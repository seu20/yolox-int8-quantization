import argparse
import time
from pathlib import Path
import runpy

import cv2
import numpy as np
import onnxruntime as ort


INPUT_SIZE = (416, 416)

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

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (255, 255, 0),
    2: (0, 255, 0),
    3: (255, 0, 255),
    5: (255, 0, 0),
    7: (0, 165, 255),
    9: (0, 0, 255),
    11: (128, 0, 255),
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


def suppress_nested_boxes(dets, threshold=0.70):
    """
    같은 클래스의 박스 중 하나가 다른 박스 내부에
    대부분 포함되어 있으면 confidence가 낮은 박스를 제거합니다.

    overlap = intersection / smaller_box_area
    """

    if dets is None or len(dets) <= 1:
        return dets

    # confidence 높은 순서
    order = np.argsort(dets[:, 4])[::-1]

    keep = []

    while len(order) > 0:
        current_idx = order[0]
        keep.append(current_idx)

        if len(order) == 1:
            break

        current = dets[current_idx]

        remaining_indices = order[1:]
        remaining = dets[remaining_indices]

        current_class = int(current[5])

        suppress = np.zeros(
            len(remaining),
            dtype=bool
        )

        for i, other in enumerate(remaining):

            # 다른 클래스는 nested suppression 하지 않음
            if int(other[5]) != current_class:
                continue

            xx1 = max(current[0], other[0])
            yy1 = max(current[1], other[1])
            xx2 = min(current[2], other[2])
            yy2 = min(current[3], other[3])

            inter_w = max(0.0, xx2 - xx1)
            inter_h = max(0.0, yy2 - yy1)

            intersection = inter_w * inter_h

            area_current = max(
                0.0,
                (current[2] - current[0])
                *
                (current[3] - current[1])
            )

            area_other = max(
                0.0,
                (other[2] - other[0])
                *
                (other[3] - other[1])
            )

            smaller_area = min(
                area_current,
                area_other
            )

            if smaller_area <= 0:
                continue

            overlap_small = (
                intersection / smaller_area
            )

            if overlap_small >= threshold:
                suppress[i] = True

        order = remaining_indices[~suppress]

    return dets[keep]


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

    # 1차: 일반 NMS
    dets = multiclass_nms(
        boxes_xyxy,
        scores,
        nms_thr=0.45,
        score_thr=0.30,
        class_agnostic=True
    )

    # 2차: nested box 제거
    dets = suppress_nested_boxes(
        dets,
        threshold=0.70
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

    # 도로 관련 클래스만 표시
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

    # 얇은 bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        1
    )

    text_y = max(
        15,
        y1 - 4
    )

    # 텍스트 검은 outline
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    # 클래스 색상 텍스트
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        required=True
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

    images = get_images(
        args.sequence
    )

    if not images:
        raise RuntimeError(
            f"No JPG/PNG images found: "
            f"{args.sequence}"
        )

    print(
        f"Images : {len(images)}"
    )

    print(
        f"Model  : {args.model}"
    )

    session = ort.InferenceSession(
        args.model,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )

    first = cv2.imread(
        str(images[0])
    )

    if first is None:
        raise RuntimeError(
            f"Failed to read image: "
            f"{images[0]}"
        )

    height, width = (
        first.shape[:2]
    )

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        args.fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "VideoWriter open failed"
        )

    latencies = []

    for i, path in enumerate(
        images,
        1
    ):

        frame = cv2.imread(
            str(path)
        )

        if frame is None:
            print(
                f"Skip: {path}"
            )
            continue

        dets, latency = detect(
            session,
            input_name,
            frame
        )

        latencies.append(
            latency
        )

        if dets is not None:

            for det in dets:

                draw_detection(
                    frame,
                    det,
                    width,
                    height
                )

        writer.write(
            frame
        )

        if i % 100 == 0:
            print(
                f"{i}/{len(images)}"
            )

    writer.release()

    if not latencies:
        raise RuntimeError(
            "No frames were processed"
        )

    avg_latency = float(
        np.mean(latencies)
    )

    avg_fps = (
        1000.0 / avg_latency
    )

    print()
    print("Finished")

    print(
        f"Frames      : "
        f"{len(latencies)}"
    )

    print(
        f"Avg Latency : "
        f"{avg_latency:.2f} ms"
    )

    print(
        f"Avg FPS     : "
        f"{avg_fps:.2f}"
    )

    print(
        f"Saved       : "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()