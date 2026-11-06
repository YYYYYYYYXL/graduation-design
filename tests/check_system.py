"""真实权重的接口冒烟检查，不训练、不改动权重或正式数据。"""
import io
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as service
from inference import model_list


def png(pixels):
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    stream.seek(0)
    return stream


def main():
    rng = np.random.default_rng(42)
    pixels = rng.integers(30, 220, (65, 73, 3), dtype=np.uint8)
    client = service.app.test_client()
    def get(url):
        response = client.get(url)
        response.get_data()
        response.close()
        return response

    original_output = service.OUTPUT
    with tempfile.TemporaryDirectory(prefix="codex_denoise_") as folder:
        service.OUTPUT = Path(folder)
        try:
            for item in model_list():
                if not item["available"]:
                    continue
                response = client.post("/api/denoise", data={
                    "model": item["id"], "image": (png(pixels), "input.png"),
                    "synthetic": "true", "sigma": "25", "seed": "42"})
                assert response.status_code == 200, (item["id"], response.json)
                result = response.json
                downloaded = get(result["output_url"] + "?download=1")
                assert downloaded.status_code == 200
                assert "attachment" in downloaded.headers["Content-Disposition"]
                image = Image.open(io.BytesIO(downloaded.data))
                assert image.size == (73, 65) and image.mode == "RGB"
                assert result["metrics"]["ssim"] is not None
                assert get(result["csv_url"]).status_code == 200
                assert get(result["json_url"]).json["model"] == item["id"]
                print(json.dumps({"model": item["id"], "status": "PASS", "device": result["device"]}, ensure_ascii=False), flush=True)
            # 实际跨块推理，确认输出没有缩放或裁掉边缘。
            large = rng.integers(0, 256, (259, 277, 3), dtype=np.uint8)
            response = client.post("/api/denoise", data={"model": "dncnn_official", "image": (png(large), "odd.png")})
            assert response.status_code == 200, response.json
            assert response.json["tiled"] and response.json["metrics"]["psnr"] is None
            assert Image.open(io.BytesIO(get(response.json["output_url"]).data)).size == (277, 259)
            # 默认模型与透明图片路径。
            rgba = np.dstack([pixels, np.full(pixels.shape[:2], 128, dtype=np.uint8)])
            response = client.post("/api/denoise", data={"image": (png(rgba), "alpha.png")})
            assert response.status_code == 200, response.json
            assert response.json["model"] == service.DEFAULT_MODEL
            cases = [
                ({"model": "missing", "image": (png(pixels), "a.png")}, 400),
                ({"model": "dncnn_sidd", "image": (png(pixels), "a.png")}, 400),
                ({"image": (io.BytesIO(b"invalid"), "a.png")}, 400),
                ({"image": (png(pixels), "a.png"), "sigma": "nan"}, 400),
                ({"image": (png(pixels), "a.png"), "seed": "1.5"}, 400),
                ({"image": (png(pixels), "a.png"), "reference": (png(pixels[:30]), "gt.png")}, 400),
                ({"image": (png(pixels), "a.png"), "synthetic": "true"}, 400),
            ]
            for data, expected in cases:
                response = client.post("/api/denoise", data=data)
                assert response.status_code == expected, response.json
            service.INFERENCE_LOCK.acquire()
            try:
                assert client.post("/api/denoise").status_code == 409
            finally:
                service.INFERENCE_LOCK.release()
            print("PASS: tiled image, default model, transparency, downloads, invalid inputs, busy handling", flush=True)
        finally:
            service.OUTPUT = original_output


if __name__ == "__main__":
    main()
