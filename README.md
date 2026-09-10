# YOLOX-Tiny INT8 Quantization on Raspberry Pi 4

YOLOX-Tiny ONNX 모델을 INT8로 양자화하고, Raspberry Pi 4 CPU에서 FP32와 INT8의 정확도와 추론 성능을 비교한 엣지 AI 경량화 프로젝트입니다.

---

## 1. 프로젝트 목표

- YOLOX-Tiny FP32 ONNX 모델을 INT8로 양자화
- Raspberry Pi 4에서 FP32 / INT8 추론 성능 비교
- 정확도 손실과 속도 향상의 trade-off 확인
- 도로 주행 영상에서 양자화 전후 탐지 결과 시각화

```text
FP32 ONNX
   ↓
Static PTQ
   ↓
INT8 ONNX
   ↓
Raspberry Pi 4
   ↓
mAP / Latency / FPS / Model Size 비교
```

---

## 2. 모델

### YOLOX-Tiny

- 입력 크기: `416 × 416`
- COCO pretrained weight 사용
- ONNX Runtime CPU inference
- FP32 / INT8 모두 동일한 전처리와 후처리 적용

YOLOX-Tiny는 연산량이 작은 경량 모델이라 Raspberry Pi와 같은 엣지 환경에서 테스트하기에 적합합니다.

다만 `416 × 416` 입력은 원거리 차량처럼 작은 객체를 탐지할 때 불리할 수 있습니다. 또한 COCO로 학습된 모델이기 때문에 KITTI 도로 환경에 특화된 fine-tuning은 적용되지 않았습니다.

---

## 3. INT8 Quantization

### 적용 방식

- 방식: Static PTQ
- Format: QDQ
- Weight Type: QInt8
- Per-channel Quantization: 적용
- Calibration: MinMax
- 입력 크기: `416 × 416`

FP32는 값을 32bit 부동소수점으로 표현하지만, INT8은 8bit 정수로 표현합니다.

```text
FP32
32 bit/value

↓ Quantization

INT8
8 bit/value
```

기대 효과:

- 모델 크기 감소
- 메모리 사용량 감소
- 메모리 대역폭 부담 감소
- 정수 연산 기반 추론 가속

대신 표현 정밀도가 줄어들기 때문에 정확도 손실이 발생할 수 있습니다.

### Calibration

Static PTQ에서는 activation 범위를 정하기 위해 calibration 데이터가 필요합니다.

```text
Calibration Images
        ↓
Activation Range 측정
        ↓
Scale / Zero Point 결정
        ↓
INT8 Model 생성
```

이번 프로젝트는 ailia-ai ONNX Quantization 예제의 calibration 이미지를 사용해 양자화 과정을 재현했습니다.

향후에는 COCO와 같이 실제 입력 분포를 더 잘 대표하는 calibration 이미지를 늘려 정확도 변화를 추가 비교할 수 있습니다.

### 실행

```bash
python quantize.py     --input_model models/yolox_tiny.opt.onnx     --output_model models/yolox_tiny_quantized_int8.onnx     --calibrate_dataset calibration_images     --per_channel True
```

---

## 4. Hardware-Specific 고려사항

### Raspberry Pi 4

- ARM64 환경
- ONNX Runtime `CPUExecutionProvider` 사용
- NPU 없이 CPU에서 FP32 / INT8 성능 직접 비교
- 평균 지연시간, P50, P95, FPS 측정

이 프로젝트에서는 단순한 모델 파일 비교가 아니라 실제 Raspberry Pi 4에서의 추론 성능을 기준으로 양자화 효과를 확인했습니다.

### PC와 Raspberry Pi 환경 분리

```text
PC
x86-64
→ 양자화 / 모델 생성

Raspberry Pi 4
ARM64
→ 추론 / 성능 측정
```

패키지 호환성이 다르기 때문에 환경을 분리했습니다.

- `requirements.txt`: PC 양자화 환경
- `requirements-rpi.txt`: Raspberry Pi 추론 환경

---

## 5. 성능 비교

COCO128 기준 FP32 / INT8 비교 결과입니다.

| 지표 | FP32 | INT8 |
|---|---:|---:|
| mAP50-95 | 43.94 | 41.98 |
| mAP50 | 62.50 | 61.38 |
| 평균 지연시간 | 312.26 ms | 144.42 ms |
| FPS | 3.20 | 6.92 |
| 모델 크기 | 19.28 MB | 5.10 MB |

### INT8 적용 결과

- 평균 지연시간: `53.8% 감소`
- FPS: `116.2% 증가`
- 모델 크기: `73.6% 감소`
- mAP50-95: `1.95%p 감소`

즉, 정확도 손실을 비교적 작게 유지하면서 Raspberry Pi 4에서 처리량을 약 2.16배 높였습니다.

> mAP는 COCO128 기준입니다. KITTI 영상은 시각적 데모 용도로 사용했습니다.

---

## 6. 도로 주행 영상 데모

KITTI 도로 주행 이미지 시퀀스를 사용해 FP32와 INT8 탐지 결과를 비교했습니다.

시각화 클래스:

- person
- bicycle
- car
- motorcycle
- bus
- truck
- traffic light
- stop sign

데모 영상은 정량 평가가 아니라 양자화 전후 탐지 결과가 시각적으로 얼마나 유지되는지 확인하는 용도입니다.

### 확인한 한계

- `416 × 416` 입력으로 인해 원거리 차량이 작게 표현됨
- COCO pretrained 모델이라 KITTI 환경에 최적화되지 않음
- 작은 객체에서 Bounding Box 위치가 불안정할 수 있음
- NMS 조정만으로는 중복 Box 개선 효과가 제한적이었음

---

## 7. 실행 환경

### PC

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Raspberry Pi 4

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-rpi.txt
```

---

## 8. Repository 구조

```text
.
├── calibration_images/
├── models/
│   ├── yolox_tiny.opt.onnx
│   └── yolox_tiny_quantized_int8.onnx
├── utils/
│   └── demo_utils.py
├── quantize.py
├── data_reader.py
├── compare_tiny.py
├── video_demo.py
├── requirements.txt
├── requirements-rpi.txt
└── README.md
```

주요 파일:

- `quantize.py`: FP32 → INT8 양자화
- `data_reader.py`: calibration 데이터 입력
- `compare_tiny.py`: FP32 / INT8 성능 비교
- `video_demo.py`: 도로 주행 영상 객체 탐지 데모

---

## 9. 개선 방향

- Calibration 데이터 확대
- KITTI 또는 도로 주행 데이터 기반 fine-tuning
- 입력 해상도 증가에 따른 정확도 / 지연시간 비교
- Raspberry Pi CPU thread 수에 따른 성능 비교
- NPU 환경에서 INT8 추가 가속 비교

---

## 참고

- YOLOX
- ONNX Runtime Quantization
- ailia-ai ONNX Quantization Example
- KITTI Vision Benchmark Suite