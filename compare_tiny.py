import os, time, cv2, runpy, io
import numpy as np
import onnxruntime as ort
from pathlib import Path
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

FP32 = "ailia_onnx_quantization/models/yolox_tiny_fp32.onnx"
INT8 = "ailia_onnx_quantization/models/yolox_tiny_int8.onnx"

IMG_DIR = Path("/workspace/int8_quantization/data/coco128/images/train2017")
LBL_DIR = Path("/workspace/int8_quantization/data/coco128/labels/train2017")

SIZE = (416, 416)

utils = runpy.run_path("utils/demo_utils.py")
demo_postprocess = utils["demo_postprocess"]
multiclass_nms = utils["multiclass_nms"]

images = sorted(IMG_DIR.glob("*.jpg"))
assert len(images) == 128
assert LBL_DIR.exists()

# COCO128 YOLO labels -> COCO ground truth
dataset = {
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{"id": i + 1, "name": str(i)} for i in range(80)],
}

ann_id = 1

for img_id, path in enumerate(images, 1):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]

    dataset["images"].append({
        "id": img_id,
        "file_name": path.name,
        "width": w,
        "height": h,
    })

    label_path = LBL_DIR / (path.stem + ".txt")

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue

            cls, xc, yc, bw, bh = map(float, v[:5])

            bw *= w
            bh *= h
            x = xc * w - bw / 2
            y = yc * h - bh / 2

            dataset["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

coco_gt = COCO()
coco_gt.dataset = dataset

with redirect_stdout(io.StringIO()):
    coco_gt.createIndex()


def preprocess(img):
    h, w = img.shape[:2]
    r = min(416 / h, 416 / w)

    resized = cv2.resize(
        img,
        (int(w * r), int(h * r)),
        interpolation=cv2.INTER_LINEAR
    )

    padded = np.full((416, 416, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized

    x = np.ascontiguousarray(
        padded.transpose(2, 0, 1),
        dtype=np.float32
    )[None]

    return x, r


def evaluate(model_path):
    sess = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name

    first = cv2.imread(str(images[0]))
    x, _ = preprocess(first)

    for _ in range(10):
        sess.run(None, {input_name: x})

    times = []
    results = []

    for img_id, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        x, ratio = preprocess(img)

        t0 = time.perf_counter()
        raw = sess.run(None, {input_name: x})[0]
        times.append((time.perf_counter() - t0) * 1000)

        pred = demo_postprocess(raw.copy(), SIZE)[0]

        boxes = pred[:, :4]
        scores = pred[:, 4:5] * pred[:, 5:]

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy /= ratio

        dets = multiclass_nms(
            xyxy,
            scores,
            nms_thr=0.65,
            score_thr=0.01,
            class_agnostic=False
        )

        if dets is None:
            continue

        for d in dets:
            x1, y1, x2, y2, score, cls = d

            x1 = max(0, min(float(x1), w))
            y1 = max(0, min(float(y1), h))
            x2 = max(0, min(float(x2), w))
            y2 = max(0, min(float(y2), h))

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            results.append({
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [x1, y1, bw, bh],
                "score": float(score),
            })

    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    t = np.array(times)

    return {
        "map": ev.stats[0] * 100,
        "map50": ev.stats[1] * 100,
        "mean": t.mean(),
        "p50": np.percentile(t, 50),
        "p95": np.percentile(t, 95),
        "fps": 1000 / t.mean(),
        "size": os.path.getsize(model_path) / 1024 / 1024,
    }


fp = evaluate(FP32)
iq = evaluate(INT8)

print("\n================ PERFORMANCE ================")
print(f"{'Metric':<15}{'FP32':>12}{'INT8':>12}")
print(f"{'mAP50-95 (%)':<15}{fp['map']:12.2f}{iq['map']:12.2f}")
print(f"{'mAP50 (%)':<15}{fp['map50']:12.2f}{iq['map50']:12.2f}")
print(f"{'Mean (ms)':<15}{fp['mean']:12.2f}{iq['mean']:12.2f}")
print(f"{'P50 (ms)':<15}{fp['p50']:12.2f}{iq['p50']:12.2f}")
print(f"{'P95 (ms)':<15}{fp['p95']:12.2f}{iq['p95']:12.2f}")
print(f"{'FPS':<15}{fp['fps']:12.2f}{iq['fps']:12.2f}")
print(f"{'Size (MB)':<15}{fp['size']:12.2f}{iq['size']:12.2f}")

print("\n================ CHANGE =====================")
print(f"mAP50-95 change : {iq['map'] - fp['map']:+.2f} point")
print(f"mAP50 change    : {iq['map50'] - fp['map50']:+.2f} point")
print(f"Latency change  : {(iq['mean']/fp['mean']-1)*100:+.1f}%")
print(f"FPS change      : {(iq['fps']/fp['fps']-1)*100:+.1f}%")
print(f"Size reduction  : {(1-iq['size']/fp['size'])*100:.1f}%")
