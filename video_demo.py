import argparse
import time
from pathlib import Path
import runpy

import cv2
import numpy as np
import onnxruntime as ort


INPUT_SIZE = (416, 416)

# COCO class ID 중 도로 주행과 관련된 클래스만 사용
ROAD_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

# OpenCV는 BGR 순서
CLASS_COLORS = {
    0: (0, 255, 255),      # person
    1: (255, 255, 0),      # bicycle
    2: (0, 255, 0),        # car
    3: (255, 0, 255),      # motorcycle
    5: (255, 0, 0),        # bus
    7: (0, 165, 255),      # truck
    9: (0, 0, 255),        # traffic light
    11: (128, 0, 255),     # stop sign
}


utils = runpy.run_path("utils/demo_utils.py")

demo_postprocess = utils["demo_postprocess"]
multiclass_nms = utils["multiclass_nms"]


def preprocess(img):
    h, w = img.shape[:2]

    ratio = min(
        INPUT_SIZE[0] / h,
        INPUT_SIZE[1] / w
    )

    resized = cv2.resize(
        img,
        (int(w * ratio), int(h * ratio)),
        interpolation=cv2.INTER_LINEAR
    )

    padded = np.full(
        (INPUT_SIZE[0], INPUT_SIZE[1], 3),
        114,
        dtype=np.uint8
    )

    padded[
        :resized.shape[0],
        :resized.shape[1]
    ] = resized

    x = padded.transpose(2, 0, 1)

    x = np.ascontiguousarray(
        x,
        dtype=np.float32
    )

    return x[None], ratio


def detect(session, input_name, frame):
    x, ratio = preprocess(frame)

    start = time.perf_counter()

    output = session.run(
        None,
        {input_name: x}
    )[0]

    latency = (
        time.perf_counter() - start
    ) * 1000.0

    pred = demo_postprocess(
        output,
        INPUT_SIZE
    )[0]

    boxes = pred[:, :4]

    scores = (
        pred[:, 4:5]
        *
        pred[:, 5:]
    )

    boxes_xyxy = np.empty_like(boxes)

    boxes_xyxy[:, 0] = (
        boxes[:, 0] - boxes[:, 2] / 2
    )

    boxes_xyxy[:, 1] = (
        boxes[:, 1] - boxes[:, 3] / 2
    )

    boxes_xyxy[:, 2] = (
        boxes[:, 0] + boxes[:, 2] / 2
    )

    boxes_xyxy[:, 3] = (
        boxes[:, 1] + boxes[:, 3] / 2
    )

    boxes_xyxy /= ratio

    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        nms_thr=0.45,
        score_thr=0.25,
        class_agnostic=False
    )

    return dets, latency


def get_images(sequence_path):
    sequence = Path(sequence_path)

    extensions = {
        ".jpg",
        ".jpeg",
        ".png"
    }

    images = sorted(
        path
        for path in sequence.iterdir()
        if path.suffix.lower() in extensions
    )

    return images


def draw_detection(frame, det, width, height):
    x1, y1, x2, y2, score, cls_id = det

    cls_id = int(cls_id)

    # 도로 관련 클래스가 아니면 표시하지 않음
    if cls_id not in ROAD_CLASSES:
        return

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))

    x2 = int(
        min(width - 1, x2)
    )

    y2 = int(
        min(height - 1, y2)
    )

    class_name = ROAD_CLASSES[cls_id]
    color = CLASS_COLORS[cls_id]

    label = (
        f"{class_name} "
        f"{score:.2f}"
    )

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # 글자 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2
    )

    label_y = max(
        y1,
        text_height + 10
    )

    # 라벨 배경
    cv2.rectangle(
        frame,
        (x1, label_y - text_height - 8),
        (x1 + text_width + 6, label_y),
        color,
        -1
    )

    # 라벨 글씨
    cv2.putText(
        frame,
        label,
        (x1 + 3, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        required=True,
        help="Directory containing sequential images"
    )

    parser.add_argument(
        "--model",
        default="models/yolox_tiny_quantized_int8.onnx"
    )

    parser.add_argument(
        "--output",
        default="road_int8.mp4"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=10.0
    )

    args = parser.parse_args()

    images = get_images(args.sequence)

    if not images:
        raise RuntimeError(
            f"No JPG/PNG images found: {args.sequence}"
        )

    print(f"Images: {len(images)}")
    print(f"Model: {args.model}")

    session = ort.InferenceSession(
        args.model,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    first = cv2.imread(
        str(images[0])
    )

    if first is None:
        raise RuntimeError(
            f"Failed to read image: {images[0]}"
        )

    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "VideoWriter open failed"
        )

    total_latency = 0.0
    processed = 0

    for i, path in enumerate(images, 1):

        frame = cv2.imread(
            str(path)
        )

        if frame is None:
            print(f"Skip: {path}")
            continue

        dets, latency = detect(
            session,
            input_name,
            frame
        )

        total_latency += latency
        processed += 1

        if dets is not None:

            for det in dets:

                draw_detection(
                    frame,
                    det,
                    width,
                    height
                )

        infer_fps = 1000.0 / latency

        cv2.putText(
            frame,
            (
                f"YOLOX-Tiny | "
                f"{latency:.1f} ms | "
                f"{infer_fps:.1f} FPS"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        if i % 100 == 0:
            print(
                f"{i}/{len(images)}"
            )

    writer.release()

    if processed == 0:
        raise RuntimeError(
            "No frames were processed"
        )

    mean_latency = (
        total_latency / processed
    )

    print()
    print("Finished")
    print(f"Frames: {processed}")

    print(
        f"Mean latency: "
        f"{mean_latency:.2f} ms"
    )

    print(
        f"Mean inference FPS: "
        f"{1000.0 / mean_latency:.2f}"
    )

    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()