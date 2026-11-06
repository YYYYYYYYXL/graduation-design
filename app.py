"""本地图像去噪服务。启动后只执行用户请求的推理，不执行训练。"""

import csv
import json
import math
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from werkzeug.exceptions import HTTPException

from denoising_metrics import calculate_psnr, calculate_ssim
from inference import DEFAULT_MODEL, MODELS, ROOT, denoise, model_list

register_heif_opener()
app = Flask(__name__, static_folder="web", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
OUTPUT = ROOT / "results/web"
INFERENCE_LOCK = threading.Lock()
MAX_PIXELS = 4_000_000
MAX_INPUT_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/models")
def models():
    return jsonify(model_list())


@app.get("/api/settings")
def settings():
    return jsonify(default_model=DEFAULT_MODEL, max_pixels=MAX_PIXELS, max_input_pixels=MAX_INPUT_PIXELS,
                   max_upload_bytes=app.config["MAX_CONTENT_LENGTH"])


@app.get("/api/progress")
def progress():
    return jsonify(json.loads((ROOT / "task_progress.json").read_text(encoding="utf-8")))


@app.get("/api/history/<model_id>")
def history(model_id):
    item = MODELS.get(model_id)
    if item is None:
        return jsonify(error="未知模型"), 404
    path = ROOT / item.get("history", "__missing__")
    if not path.is_file():
        return jsonify(rows=[], note="无本地训练日志；公开预训练模型不生成本项目训练记录。")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return jsonify(rows=rows, note="原有 CSV 日志，未补写、未伪造。")


def read_image(upload):
    if upload is None or not upload.filename:
        raise ValueError("请选择图片")
    try:
        with Image.open(upload.stream) as image:
            if image.format not in ("PNG", "JPEG", "WEBP", "BMP", "HEIF"):
                raise ValueError("仅支持 PNG、JPEG、WebP、BMP、HEIC 和 HEIF")
            if image.width * image.height > MAX_INPUT_PIXELS or min(image.size) < 11:
                raise ValueError("图片每边至少 11 像素，原图总像素不能超过 1 亿")
            image = ImageOps.exif_transpose(image)
            original_size = image.size
            if image.width * image.height > MAX_PIXELS:
                scale = math.sqrt(MAX_PIXELS / (image.width * image.height))
                size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
                if min(size) < 11:
                    raise ValueError("图片过于狭长，缩小后无法处理，请先裁剪")
                image = image.resize(size, Image.Resampling.LANCZOS)
            # 透明图像统一合成至白底，避免透明区域变黑。
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                image = Image.alpha_composite(background, rgba)
            return np.array(image.convert("RGB")), original_size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取图片或图片尺寸过大") from exc


def number(name, default, low, high):
    raw = request.form.get(name, "").strip()
    try:
        value = default if not raw else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须填写数字") from exc
    if value is not None and (not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} 必须在 {low}～{high} 之间")
    return value


def metrics(image, reference):
    if reference is None:
        return dict(psnr=None, ssim=None)
    # 相同图像 PSNR 为无穷大，避免把 JSON 非有限数字或人为上限当成实测值。
    psnr = "∞" if np.array_equal(image, reference) else calculate_psnr(image, reference)
    return dict(psnr=psnr, ssim=calculate_ssim(image, reference))


@app.post("/api/denoise")
def process():
    if not INFERENCE_LOCK.acquire(blocking=False):
        return jsonify(error="另一个去噪请求正在处理，请稍后重试"), 409
    try:
        started = time.perf_counter()
        model_id = request.form.get("model", DEFAULT_MODEL)
        if model_id not in MODELS:
            raise ValueError("请选择有效模型")
        item = MODELS[model_id]
        if item["path"] and not (ROOT / item["path"]).is_file():
            raise ValueError("该模型的权重文件不存在，请选择其他模型")
        device = request.form.get("device", "auto")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("无效的运行设备")
        pixels, original_size = read_image(request.files.get("image"))
        reference = None
        if request.files.get("reference"):
            reference, reference_size = read_image(request.files["reference"])
            if reference_size != original_size:
                raise ValueError("参考图与输入原图尺寸必须相同且内容对齐")
        sigma = number("sigma", None, 0, 255)
        seed = number("seed", 42, 0, 2**32-1)
        if seed != int(seed):
            raise ValueError("seed 必须为整数")
        synthetic = request.form.get("synthetic", "false") == "true"
        if synthetic:
            if reference is not None:
                raise ValueError("合成噪声模式使用上传原图作为 GT，不要额外上传参考图")
            if sigma is None:
                raise ValueError("合成噪声需要填写 sigma")
            reference = pixels.copy()
            noise = np.random.default_rng(int(seed)).normal(0, sigma, pixels.shape)
            pixels = np.rint(np.clip(pixels.astype(np.float32) + noise, 0, 255)).astype(np.uint8)
        if reference is not None and reference.shape != pixels.shape:
            raise ValueError("参考图与输入图尺寸必须相同且内容对齐")
        run_id = uuid.uuid4().hex
        directory = OUTPUT / run_id
        directory.mkdir(parents=True)
        Image.fromarray(pixels).save(directory / "input.png")
        if reference is not None:
            Image.fromarray(reference).save(directory / "reference.png")
        timing = denoise(model_id, directory / "input.png", directory / "output.png", sigma,
                         device)
        with Image.open(directory / "output.png") as image:
            output = np.array(image)
        result = dict(id=run_id, model=model_id, model_name=MODELS[model_id]["name"],
                      created_at=datetime.now(timezone.utc).isoformat(), width=pixels.shape[1],
                      height=pixels.shape[0], original_width=original_size[0], original_height=original_size[1],
                      resized=original_size != (pixels.shape[1], pixels.shape[0]),
                      synthetic=synthetic, seed=int(seed) if synthetic else None,
                      metrics=metrics(output, reference), input_metrics=metrics(pixels, reference),
                      reference_provided=reference is not None, **timing)
        result["total_seconds"] = time.perf_counter() - started
        result["note"] = MODELS[model_id]["note"]
        result["input_url"] = f"/results/{run_id}/input.png"
        result["output_url"] = f"/results/{run_id}/output.png"
        result["reference_url"] = f"/results/{run_id}/reference.png" if reference is not None else None
        result["json_url"] = f"/results/{run_id}/result.json"
        result["csv_url"] = f"/results/{run_id}/metrics.csv"
        (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        row = {key: result[key] for key in ("id", "model", "created_at", "width", "height", "original_width", "original_height", "resized", "synthetic", "seed", "device", "seconds", "load_seconds", "total_seconds", "sigma", "tiled")}
        row.update(result["metrics"])
        with (directory / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return jsonify(result)
    finally:
        INFERENCE_LOCK.release()


@app.get("/results/<run_id>/<filename>")
def result_file(run_id, filename):
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        return jsonify(error="无效结果编号"), 404
    if filename not in ("input.png", "output.png", "reference.png", "result.json", "metrics.csv"):
        return jsonify(error="文件不存在"), 404
    return send_from_directory(OUTPUT / run_id, filename,
                               as_attachment=request.args.get("download") == "1" or filename.endswith((".csv", ".json")),
                               download_name="denoised.png" if filename == "output.png" else filename)


@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, HTTPException):
        message = "上传文件合计不能超过 50 MB" if error.code == 413 else error.description
        return jsonify(error=message), error.code
    if isinstance(error, ValueError):
        return jsonify(error=str(error)), 400
    app.logger.exception("去噪请求失败")
    return jsonify(error="处理失败，请查看服务端日志。检查依赖、权重兼容性或改用 CPU / 较小图片。"), 500


if __name__ == "__main__":
    from waitress import serve
    print("图像去噪系统：http://127.0.0.1:5000", flush=True)
    serve(app, host="127.0.0.1", port=5000, threads=4, max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
