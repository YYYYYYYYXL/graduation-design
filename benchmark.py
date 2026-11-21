"""批量评测项目中成对的固定测试图，并保留逐图结果。"""

import csv
import io
import json
import logging
import math
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
from flask import Blueprint, Response, abort, jsonify, redirect, request, send_file, url_for
from PIL import Image

from denoising_metrics import reference_scores
from inference import INFERENCE_LOCK, ROOT, denoise, model_list


benchmark = Blueprint("benchmark", __name__)
DATA = ROOT / "data"
OUTPUT = ROOT / "results/benchmark"
GROUPS = {
    f"div2k_{sigma}": (f"DIV2K · σ={sigma}", f"div2k_test/noise/sigma_{sigma}", "div2k_test/gt", sigma)
    for sigma in (15, 25, 50)
}
GROUPS["sidd"] = ("SIDD · 真实噪声", "sidd_test/noisy", "sidd_test/gt", None)
RUN_LOCK = threading.Lock()
active_run = None
PAGE_SIZE = 25

SCHEMA = """
CREATE TABLE run (created TEXT, settings TEXT, state TEXT, error TEXT);
CREATE TABLE sample (
    id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, filename TEXT NOT NULL,
    psnr REAL, ssim REAL, error TEXT
);
CREATE TABLE prediction (
    sample_id INTEGER NOT NULL, model TEXT NOT NULL,
    psnr REAL, ssim REAL, seconds REAL, device TEXT, sigma REAL, error TEXT, parameters INTEGER,
    PRIMARY KEY (sample_id, model)
);
"""


def png_names(relative):
    return {path.name for path in (DATA / relative).glob("*.png") if path.is_file()}


def paired_names(dataset):
    label, noisy, gt, _ = GROUPS[dataset]
    inputs, references = png_names(noisy), png_names(gt)
    if not inputs or inputs != references:
        raise ValueError(f"{label} 的测试图片为空或与参考图的文件名不匹配。")
    return sorted(inputs)


def connect(directory, filename="report.sqlite3"):
    db = sqlite3.connect(directory / filename)
    db.row_factory = sqlite3.Row
    return closing(db)


def find_run(run_id):
    directory = OUTPUT / run_id
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        abort(404)
    if not (directory / "report.sqlite3").is_file():
        abort(404)
    return directory


def run_info(directory, db):
    row = db.execute("SELECT * FROM run").fetchone()
    info = dict(row)
    info.update(json.loads(info.pop("settings")), id=directory.name)
    if info["state"] == "running" and active_run != directory.name:
        info.update(state="interrupted", error="服务已重启，已保存的结果可继续查看。")
    return info


def pixels(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def source_paths(sample):
    _, noisy, gt, sigma = GROUPS[sample["dataset"]]
    filename = sample["filename"]
    return DATA / noisy / filename, DATA / gt / filename, sigma


def prediction_path(directory, sample, model):
    return directory / sample["dataset"] / model / sample["filename"]


def run_batch(directory, models):
    global active_run
    try:
        with connect(directory) as db:
            samples = db.execute("SELECT * FROM sample ORDER BY id").fetchall()
            for sample in samples:
                noisy, gt, _ = source_paths(sample)
                try:
                    psnr, ssim = reference_scores(pixels(noisy), pixels(gt))
                    error = None
                except (OSError, ValueError) as exc:
                    psnr, ssim, error = None, None, str(exc)
                with db:
                    db.execute("UPDATE sample SET psnr=?, ssim=?, error=? WHERE id=?",
                               (psnr, ssim, error, sample["id"]))

            samples = db.execute("SELECT * FROM sample ORDER BY id").fetchall()
            for model in models:
                for sample in samples:
                    result = evaluate_image(directory, sample, model)
                    with db:
                        db.execute("INSERT INTO prediction VALUES (?,?,?,?,?,?,?,?,?)",
                                   (sample["id"], model["id"], *result))
            state = "complete_with_errors" if db.execute(
                "SELECT 1 FROM prediction WHERE error IS NOT NULL LIMIT 1"
            ).fetchone() else "complete"
            with db:
                db.execute("UPDATE run SET state=?", (state,))
    except Exception:
        logging.exception("批量评测中断")
        with connect(directory) as db, db:
            db.execute("UPDATE run SET state='failed', error=?", ("评测中断，请查看服务日志。",))
    finally:
        active_run = None
        RUN_LOCK.release()


def evaluate_image(directory, sample, model):
    if sample["error"] is not None:
        return None, None, None, None, None, sample["error"], None
    noisy, gt, sigma = source_paths(sample)
    output = prediction_path(directory, sample, model["id"])
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with INFERENCE_LOCK:
            timing = denoise(model["id"], noisy, output, sigma)
        psnr, ssim = reference_scores(pixels(output), pixels(gt))
        return psnr, ssim, timing["seconds"], timing["device"], timing["sigma"], None, timing["parameters"]
    except Exception as exc:
        logging.warning("批量评测失败：%s %s/%s", model["id"], sample["dataset"],
                        sample["filename"], exc_info=True)
        return None, None, None, None, None, str(exc) or type(exc).__name__, None


@benchmark.get("/benchmark")
def page():
    return redirect(url_for("compare_models", run=request.args.get("run")))


@benchmark.get("/api/benchmark/datasets")
def datasets():
    groups = []
    for dataset, (label, _, _, _) in GROUPS.items():
        try:
            count, error = len(paired_names(dataset)), None
        except ValueError as exc:
            count, error = 0, str(exc)
        groups.append(dict(id=dataset, label=label, count=count, error=error))
    return jsonify(groups)


@benchmark.get("/api/benchmark/runs")
def recent_runs():
    paths = sorted(OUTPUT.glob("*/report.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for path in paths[:15]:
        with connect(path.parent) as db:
            runs.append(run_info(path.parent, db))
    return jsonify(runs)


@benchmark.post("/api/benchmark/runs")
def start_run():
    global active_run
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("请选择测试组和模型。")
    groups, selected = body.get("datasets"), body.get("models")
    available = {m["id"]: m for m in model_list() if m["available"]}
    for values, allowed in ((groups, GROUPS), (selected, available)):
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) or value not in allowed for value in values
        ):
            raise ValueError("请至少选择一个可用测试组和一个可用模型。")
    groups = list(dict.fromkeys(groups))
    models = [dict(id=m, name=available[m]["name"], note=available[m]["note"],
                   weights_bytes=available[m]["weights_bytes"])
              for m in dict.fromkeys(selected)]
    preview = body.get("preview", False)
    if not isinstance(preview, bool):
        raise ValueError("预览模式参数无效。")
    pairs = [(group, filename) for group in groups
             for filename in paired_names(group)[:1 if preview else None]]
    if not RUN_LOCK.acquire(blocking=False):
        return jsonify(error="已有批量任务在运行。", id=active_run), 409
    directory = OUTPUT / uuid.uuid4().hex
    try:
        directory.mkdir(parents=True)
        settings = json.dumps(dict(datasets=groups, models=models, preview=preview), ensure_ascii=False)
        created = datetime.now(timezone.utc).isoformat()
        with connect(directory, "report.part") as db, db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO run VALUES (?,?,'running',NULL)", (created, settings))
            db.executemany("INSERT INTO sample (dataset, filename) VALUES (?,?)", pairs)
        (directory / "report.part").replace(directory / "report.sqlite3")
        active_run = directory.name
        threading.Thread(target=run_batch, args=(directory, models), daemon=True).start()
    except BaseException:
        try:
            if active_run == directory.name:
                active_run = None
                with connect(directory) as db, db:
                    db.execute("UPDATE run SET state='failed', error=?", ("无法启动评测。",))
        finally:
            RUN_LOCK.release()
        raise
    return jsonify(id=directory.name), 202


# 只用每个已选模型都成功的样本做横向汇总，避免对照集合不同。
COMMON = """
WITH common AS (
    SELECT sample_id FROM prediction WHERE error IS NULL
    GROUP BY sample_id HAVING COUNT(*) = ?
)
"""


def wire_value(value):
    return "∞" if isinstance(value, float) and math.isinf(value) else value


def wire_row(row):
    return {key: wire_value(value) for key, value in dict(row).items()}


def summarize(db, info):
    model_count = len(info["models"])
    baselines = {row["dataset"]: row for row in db.execute(COMMON + """
        SELECT dataset, COUNT(*) AS compared, AVG(psnr) AS psnr, AVG(ssim) AS ssim
        FROM sample JOIN common ON common.sample_id=sample.id GROUP BY dataset
    """, (model_count,))}
    # Legacy reports predate parameter-count recording.
    columns = {row["name"] for row in db.execute("PRAGMA table_info(prediction)")}
    parameters = "MAX(p.parameters)" if "parameters" in columns else "NULL"
    averages = {(row["dataset"], row["model"]): row for row in db.execute(COMMON + f"""
        SELECT dataset, model, AVG(p.psnr) AS psnr, AVG(p.ssim) AS ssim,
               AVG(seconds) AS seconds, GROUP_CONCAT(DISTINCT device) AS device, {parameters} AS parameters
        FROM sample s JOIN common ON common.sample_id=s.id
        JOIN prediction p ON p.sample_id=s.id GROUP BY dataset, model
    """, (model_count,))}
    counts = {(row["dataset"], row["model"]): row for row in db.execute("""
        SELECT dataset, model, COUNT(*) AS done, SUM(p.error IS NOT NULL) AS failed
        FROM prediction p JOIN sample s ON s.id=p.sample_id GROUP BY dataset, model
    """)}
    totals = dict(db.execute("SELECT dataset, COUNT(*) FROM sample GROUP BY dataset"))
    rows = []
    for dataset in info["datasets"]:
        baseline = baselines.get(dataset)
        base = dict(dataset=dataset, label=GROUPS[dataset][0], total=totals[dataset],
                    compared=baseline["compared"] if baseline else 0)
        rows.append(dict(base, model="input", name="去噪前", psnr=wire_value(baseline["psnr"]) if baseline else None,
                         ssim=baseline["ssim"] if baseline else None))
        for model in info["models"]:
            key = dataset, model["id"]
            avg, count = averages.get(key), counts.get(key)
            row = dict(base, model=model["id"], name=model["name"], done=count["done"] if count else 0,
                       failed=count["failed"] if count else 0, psnr=None, ssim=None, seconds=None,
                       device=None, parameters=None, weights_bytes=model.get("weights_bytes"), delta_psnr=None, delta_ssim=None)
            if avg:
                row.update({key: wire_value(avg[key]) for key in ("psnr", "ssim", "seconds", "device", "parameters")})
                if math.isfinite(avg["psnr"]) and math.isfinite(baseline["psnr"]):
                    row["delta_psnr"] = avg["psnr"] - baseline["psnr"]
                row["delta_ssim"] = avg["ssim"] - baseline["ssim"]
            rows.append(row)
    return rows


@benchmark.get("/api/benchmark/runs/<run_id>")
def report(run_id):
    directory = find_run(run_id)
    with connect(directory) as db:
        db.execute("BEGIN")
        info = run_info(directory, db)
        info["rows"] = summarize(db, info)
    return jsonify(info)


@benchmark.get("/api/benchmark/runs/<run_id>/samples")
def sample_list(run_id):
    directory = find_run(run_id)
    dataset = request.args.get("dataset", "")
    search = request.args.get("search", "").strip()[:128]
    page_number = max(1, request.args.get("page", 1, type=int))
    where = "WHERE (?='' OR dataset=?) AND INSTR(LOWER(filename), LOWER(?)) > 0"
    params = dataset, dataset, search
    with connect(directory) as db:
        db.execute("BEGIN")
        count = db.execute(f"SELECT COUNT(*) FROM sample {where}", params).fetchone()[0]
        page_number = min(page_number, max(1, math.ceil(count / PAGE_SIZE)))
        rows = db.execute(f"""
            SELECT s.*, (SELECT COUNT(*) FROM prediction WHERE sample_id=s.id AND error IS NULL) AS ready
            FROM sample s {where} ORDER BY id LIMIT ? OFFSET ?
        """, (*params, PAGE_SIZE, (page_number - 1) * PAGE_SIZE))
        items = [wire_row(row) for row in rows]
    return jsonify(items=items, total=count, page=page_number, page_size=PAGE_SIZE)


@benchmark.get("/api/benchmark/runs/<run_id>/samples/<int:sample_id>")
def sample_detail(run_id, sample_id):
    directory = find_run(run_id)
    with connect(directory) as db:
        db.execute("BEGIN")
        sample = db.execute("SELECT * FROM sample WHERE id=?", (sample_id,)).fetchone()
        if sample is None:
            abort(404)
        info = run_info(directory, db)
        predictions = {row["model"]: wire_row(row) for row in db.execute(
            "SELECT * FROM prediction WHERE sample_id=?", (sample_id,)
        )}
    url = f"/api/benchmark/runs/{run_id}/samples/{sample_id}/images/"
    return jsonify(
        sample=wire_row(sample), label=GROUPS[sample["dataset"]][0], input_url=url + "input",
        reference_url=url + "reference",
        models=[dict(model, result=predictions.get(model["id"]),
                     url=url + model["id"] if model["id"] in predictions
                     and predictions[model["id"]]["error"] is None else None)
                for model in info["models"]],
    )


@benchmark.get("/api/benchmark/runs/<run_id>/samples/<int:sample_id>/images/<kind>")
def sample_image(run_id, sample_id, kind):
    directory = find_run(run_id)
    with connect(directory) as db:
        sample = db.execute("SELECT * FROM sample WHERE id=?", (sample_id,)).fetchone()
        if sample is None:
            abort(404)
        if kind in ("input", "reference"):
            noisy, gt, _ = source_paths(sample)
            path = noisy if kind == "input" else gt
        elif db.execute("SELECT 1 FROM prediction WHERE sample_id=? AND model=? AND error IS NULL",
                        (sample_id, kind)).fetchone():
            path = prediction_path(directory, sample, kind)
        else:
            abort(404)
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/png")


@benchmark.get("/api/benchmark/runs/<run_id>/<report_type>.csv")
def download_report(run_id, report_type):
    directory = find_run(run_id)
    with connect(directory) as db:
        db.execute("BEGIN")
        info = run_info(directory, db)
        if report_type == "summary":
            fields = ("dataset", "label", "model", "name", "total", "compared", "done", "failed",
                      "psnr", "ssim", "delta_psnr", "delta_ssim", "seconds", "device", "parameters", "weights_bytes")
            rows = summarize(db, info)
        elif report_type == "images":
            fields = ("dataset", "filename", "model", "input_psnr", "input_ssim", "psnr", "ssim",
                      "seconds", "device", "sigma", "error")
            rows = [wire_row(row) for row in db.execute("""
                SELECT dataset, filename, model, s.psnr AS input_psnr, s.ssim AS input_ssim,
                       p.psnr, p.ssim, seconds, device, sigma, p.error
                FROM sample s JOIN prediction p ON p.sample_id=s.id ORDER BY s.id, model
            """)]
        else:
            abort(404)
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=(*fields, "preview", "run_state"))
    writer.writeheader()
    writer.writerows(dict(row, preview=info["preview"], run_state=info["state"]) for row in rows)
    return Response("\ufeff" + content.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=benchmark_{report_type}.csv"})
