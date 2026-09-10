import os
import time
import io
import cv2
import runpy
import unicodedata
import numpy as np
import onnxruntime as ort

from pathlib import Path
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


FP32 = "models/yolox_tiny.opt.onnx"
INT8 = "models/yolox_tiny_quantized_int8.onnx"

IMG_DIR = Path("/workspace/int8_quantization/data/coco128/images/train2017")
LBL_DIR = Path("/workspace/int8_quantization/data/coco128/labels/train2017")

SIZE = (416, 416)


# YOLOX 후처리 함수 불러오기
utils = runpy.run_path("utils/demo_utils.py")
demo_postprocess = utils["demo_postprocess"]
multiclass_nms = utils["multiclass_nms"]


# COCO128 이미지 확인
images = sorted(IMG_DIR.glob("*.jpg"))

assert len(images) == 128
assert LBL_DIR.exists()


# =========================================================
# COCO128의 YOLO 라벨을 COCO 평가 형식으로 변환
# =========================================================

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

    label_path = LBL_DIR / f"{path.stem}.txt"

    if not label_path.exists():
        continue

    for line in label_path.read_text().splitlines():
        values = line.split()

        if len(values) < 5:
            continue

        cls, xc, yc, bw, bh = map(float, values[:5])

        # 정규화된 YOLO 좌표를 실제 픽셀 좌표로 변환
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


# COCO 평가용 정답 데이터 생성
coco_gt = COCO()
coco_gt.dataset = dataset

with redirect_stdout(io.StringIO()):
    coco_gt.createIndex()


# =========================================================
# YOLOX 입력 전처리
# =========================================================

def preprocess(img):
    h, w = img.shape[:2]

    # 종횡비를 유지하면서 416x416 내부에 맞춤
    ratio = min(416 / h, 416 / w)

    resized = cv2.resize(
        img,
        (int(w * ratio), int(h * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    # 남는 영역은 YOLOX 기본 padding 값 114 사용
    padded = np.full((416, 416, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized

    # HWC -> CHW 변환 후 batch 차원 추가
    x = np.ascontiguousarray(
        padded.transpose(2, 0, 1),
        dtype=np.float32,
    )[None]

    return x, ratio


# =========================================================
# 모델 성능 평가
# =========================================================

def evaluate(model_path):
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # 초기 실행 비용을 제외하기 위한 워밍업
    first = cv2.imread(str(images[0]))
    x, _ = preprocess(first)

    for _ in range(10):
        session.run(None, {input_name: x})

    times = []
    results = []

    for img_id, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]

        x, ratio = preprocess(img)

        # 추론 시간 측정
        start = time.perf_counter()

        raw = session.run(
            None,
            {input_name: x},
        )[0]

        latency_ms = (time.perf_counter() - start) * 1000
        times.append(latency_ms)

        # YOLOX 출력 후처리
        pred = demo_postprocess(raw.copy(), SIZE)[0]

        boxes = pred[:, :4]

        # 객체 존재 확률 × 클래스 확률
        scores = pred[:, 4:5] * pred[:, 5:]

        # 중심 좌표 형식을 x1, y1, x2, y2 형식으로 변환
        xyxy = np.empty_like(boxes)

        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        # 원본 이미지 크기에 맞게 좌표 복원
        xyxy /= ratio

        # 중복 Bounding Box 제거
        dets = multiclass_nms(
            xyxy,
            scores,
            nms_thr=0.65,
            score_thr=0.01,
            class_agnostic=False,
        )

        if dets is None:
            continue

        for det in dets:
            x1, y1, x2, y2, score, cls = det

            # Bounding Box가 이미지 영역을 벗어나지 않도록 제한
            x1 = max(0, min(float(x1), w))
            y1 = max(0, min(float(y1), h))
            x2 = max(0, min(float(x2), w))
            y2 = max(0, min(float(y2), h))

            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                continue

            # COCO 평가 형식으로 탐지 결과 저장
            results.append({
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [x1, y1, bw, bh],
                "score": float(score),
            })

    # COCO 방식으로 mAP 계산
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)

        evaluator = COCOeval(
            coco_gt,
            coco_dt,
            "bbox",
        )

        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    times = np.array(times)

    return {
        "map": evaluator.stats[0] * 100,
        "map50": evaluator.stats[1] * 100,
        "mean": times.mean(),
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "fps": 1000.0 / times.mean(),
        "size": os.path.getsize(model_path) / 1024 / 1024,
    }


# =========================================================
# 한글 터미널 정렬
# =========================================================

def display_width(text):
    width = 0

    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1

    return width


def pad(text, width):
    return text + " " * max(0, width - display_width(text))


# =========================================================
# FP32 / INT8 평가
# =========================================================

fp = evaluate(FP32)
iq = evaluate(INT8)


# =========================================================
# 성능 비교 출력
# =========================================================

LABEL_WIDTH = 28
VALUE_WIDTH = 12

print("\n================ 성능 비교 ================")

print(
    pad("지표", LABEL_WIDTH)
    + f"{'FP32':>{VALUE_WIDTH}}"
    + f"{'INT8':>{VALUE_WIDTH}}"
)

print(
    pad("mAP50-95 (%)", LABEL_WIDTH)
    + f"{fp['map']:>{VALUE_WIDTH}.2f}"
    + f"{iq['map']:>{VALUE_WIDTH}.2f}"
)

print(
    pad("mAP50 (%)", LABEL_WIDTH)
    + f"{fp['map50']:>{VALUE_WIDTH}.2f}"
    + f"{iq['map50']:>{VALUE_WIDTH}.2f}"
)

print(
    pad("평균 지연시간 (ms)", LABEL_WIDTH)
    + f"{fp['mean']:>{VALUE_WIDTH}.2f}"
    + f"{iq['mean']:>{VALUE_WIDTH}.2f}"
)

print(
    pad("초당 처리 프레임 (FPS)", LABEL_WIDTH)
    + f"{fp['fps']:>{VALUE_WIDTH}.2f}"
    + f"{iq['fps']:>{VALUE_WIDTH}.2f}"
)

print(
    pad("모델 크기 (MB)", LABEL_WIDTH)
    + f"{fp['size']:>{VALUE_WIDTH}.2f}"
    + f"{iq['size']:>{VALUE_WIDTH}.2f}"
)


# =========================================================
# 변화량 계산 및 출력
# =========================================================

map_change = iq["map"] - fp["map"]
map50_change = iq["map50"] - fp["map50"]
latency_change = (iq["mean"] / fp["mean"] - 1) * 100
fps_change = (iq["fps"] / fp["fps"] - 1) * 100
size_reduction = (1 - iq["size"] / fp["size"]) * 100


print("\n================ 변화량 ====================")

print(
    pad("mAP50-95 변화", 24)
    + f": {map_change:+.2f} %p"
)

print(
    pad("mAP50 변화", 24)
    + f": {map50_change:+.2f} %p"
)

print(
    pad("평균 지연시간 변화", 24)
    + f": {latency_change:+.1f}%"
)

print(
    pad("FPS 변화", 24)
    + f": {fps_change:+.1f}%"
)

print(
    pad("모델 크기 감소율", 24)
    + f": {size_reduction:.1f}%"
)