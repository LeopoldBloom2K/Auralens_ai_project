# AuraLens AI Pipeline

MobileNetV3-Large 장면 분류 모델을 학습시키고 Flutter 앱에 배포하는 전체 파이프라인입니다.

---

## 개요

```
[커스텀 데이터 정리]
       ↓
[Kaggle 데이터셋 다운로드]
       ↓
[prepare_dataset.py] — 학습/검증 폴더 분리
       ↓
[train.py] — MobileNetV3 전이학습 (person/food/scenery)
       ↓
[export_tflite.py] — PyTorch → ONNX 변환 (IR version 9 패치 포함)
       ↓
[convert_to_tflite.py] — ONNX → TFLite 동적 범위 양자화
       ↓
[evaluate_models.py] — ONNX vs TFLite 정확도 비교
       ↓
[Flutter 앱] — ONNX Runtime 장면분류 + ML Kit 객체감지
               피보나치 구도 오버레이 + 터치·자동 포커스
```

---

## 1단계: 커스텀 데이터 정리 (선택 단계)

**스크립트:** `process_custom_data.py`

> **주의:** 현재 커스텀 데이터는 person≈30장·food≈60장·scenery≈60장으로 Kaggle 데이터(클래스당 1,668~5,000장)의 1~2%에 불과합니다. 배치 1개 미만 수량은 WeightedRandomSampler의 균등 샘플링 과정에서 희석되어 학습 결과에 실질적인 영향을 주기 어렵습니다. **클래스당 200장 이상을 확보하기 전까지는 이 단계를 건너뛰어도 무방합니다.**

`./data/` 폴더의 KakaoTalk 이미지를 타임스탬프 그룹별로 분류하여 각 raw dataset 폴더에 복사합니다.

**분류 로직:** 각 `.jpg` 파일의 `stem`(확장자 제외 이름)이 지정된 접두사로 시작하는지 `str.startswith()`로 판별합니다. 매칭되지 않은 파일은 스킵됩니다.

| 그룹 접두사                     | 클래스  | 복사 대상 폴더                | 설명           |
| ------------------------------- | ------- | ----------------------------- | -------------- |
| KakaoTalk*20260621_000014850*\* | person  | `gender_dataset/men/custom/`  | 인물+풍경 혼합 |
| KakaoTalk*20260621_000030313*\* | scenery | `seg_train/seg_train/custom/` | 실내·도시 공간 |
| KakaoTalk*20260621_000057355*\* | food    | `food11/training/custom/`     | 음식 위주      |
| KakaoTalk*20260621_000043833*\* | food    | `food11/training/custom/`     | 음료·음식      |
| KakaoTalk*20260621_000337932*\* | scenery | `seg_train/seg_train/custom/` | 자연·건축 경관 |

각 `custom/` 서브폴더는 `prepare_dataset.py`의 `rglob` 탐색 경로 하위에 위치하여 Kaggle 데이터와 함께 자동으로 수집됩니다.

```bash
python process_custom_data.py
```

---

## 2단계: 데이터셋 준비

**스크립트:** `prepare_dataset.py`

Kaggle에서 다운받은 raw dataset을 `./dataset/train/` 및 `./dataset/val/` 구조로 복사합니다.

**사용 데이터셋:**

- `snmahsa/human-images-dataset-men-and-women` → **person**
- `vermaavi/food11` → **food**
- `puneet6060/intel-image-classification` → **scenery**

**클래스 3개 (unknown 제거):**

```python
classes = ['person', 'food', 'scenery']
```

```bash
python prepare_dataset.py
```

결과 구조:

```
dataset/
  train/
    food/
    person/
    scenery/
  val/
    food/
    person/
    scenery/
```

---

## 3단계: 모델 학습

**스크립트:** `train.py`

MobileNetV3-Large (ImageNet 사전학습) 의 마지막 분류 레이어를 3-class Linear로 교체하여 전이학습합니다.

**클래스 불균형 대응 — WeightedRandomSampler:**

person 데이터(~1,668장)가 food/scenery(~5,000장)보다 적으므로 배치마다 클래스 비율이 균등하도록 샘플링합니다.

```python
class_counts = [targets.count(i) for i in range(len(class_names))]
sample_weights = [1.0 / class_counts[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
```

**데이터 증강:**

| 단계 | 변환 |
|------|------|
| train | RandomResizedCrop(224) + RandomHorizontalFlip + ColorJitter |
| val   | Resize(256) → CenterCrop(224) |

ImageNet 정규화: mean=`[0.485, 0.456, 0.406]`, std=`[0.229, 0.224, 0.225]`

**주요 하이퍼파라미터:**

| 항목 | 값 |
|---|---|
| 모델 | MobileNetV3-Large (IMAGENET1K_V2) |
| 배치 크기 | 128 |
| 에포크 | 10 |
| 옵티마이저 | Adam (lr=0.001) |
| 손실함수 | CrossEntropyLoss |
| num_workers | 4 (DataLoader 병렬 로딩) |

최고 검증 정확도 기록 시 `./weights/best_auralens_model.pth` 자동 저장.

```bash
python train.py
```

---

## 4단계: ONNX 변환

**스크립트:** `export_tflite.py`

학습된 `.pth` 모델을 ONNX로 변환합니다. 이 단계에서 클래스 순서를 앱 기대값에 맞게 재배치합니다.

**클래스 순서 문제:**

`ImageFolder`는 알파벳 순서로 클래스를 정렬합니다 (food=0, person=1, scenery=2).
앱 코드는 person=0, food=1, scenery=2 순서를 기대합니다.

이를 `_AppOrderWrapper`로 해결합니다:

```python
class _AppOrderWrapper(nn.Module):
    def forward(self, x):
        out = self.base(x)
        # food→1, person→0, scenery→2 로 재배치
        return torch.stack([out[:, 1], out[:, 0], out[:, 2]], dim=1)
```

재배치 연산은 `torch.onnx.export()` 시점에 ONNX 그래프 노드로 직접 구워집니다. 따라서 5단계(`convert_to_tflite.py`)에서 별도 처리 없이 TFLite 모델에도 순서가 그대로 유지됩니다. 클래스 인덱스 흐름 요약:

```
ImageFolder 학습:   food=0  person=1  scenery=2  (알파벳 순)
_AppOrderWrapper:   person=0  food=1  scenery=2  (ONNX에 베이킹)
TFLite 변환 후:     person=0  food=1  scenery=2  ← 앱 기대값과 일치
inference_service:  index 0 → SceneCategory.person  ✅
                    index 1 → SceneCategory.food     ✅
                    index 2 → SceneCategory.scenery  ✅
```

**AdaptiveAvgPool → 고정 크기 AvgPool 교체:**

ONNX 내보내기 전 `model.avgpool`을 `nn.AvgPool2d(kernel_size=7, stride=1)`으로 교체합니다. 224×224 입력의 MobileNetV3 특징 맵이 7×7이므로 동적 풀링 없이도 동일 결과를 내며, ONNX 그래프의 동적 형상 노드를 제거해 변환 안정성을 높입니다.

**수정된 버그:**

- `classifier[4]` → `classifier[3]` (MobileNetV3 구조에 맞는 인덱스)
- `opset_version=11` 명시
- `dynamic_axes` 추가로 배치 크기 유연화

```bash
python export_tflite.py
```

출력: `./weights/auralens_model.onnx`

---

## 5단계: TFLite 동적 범위 양자화 변환

**스크립트:** `convert_to_tflite.py`

ONNX 모델을 `onnx2tf`로 TensorFlow SavedModel로 변환한 뒤, 동적 범위 양자화를 적용하여 TFLite로 압축합니다.

**0단계 — ONNX 그래프 단순화 (`onnxsim`):**

변환 전 `onnxsim`으로 RESHAPE 등 복잡한 노드를 정리합니다. 단순화 실패 시 원본 ONNX를 그대로 사용하며 변환을 계속합니다.

```python
simplified, ok = onnxsim.simplify(model)
```

**1단계 — ONNX → TF SavedModel (`onnx2tf`):**

```python
onnx2tf.convert(
    input_onnx_file_path=src_onnx,
    output_folder_path=tf_model_path,
    copy_onnx_input_output_names_to_tflite=True,
)
```

이전 변환 캐시(`./weights/tf_saved_model/`)를 삭제한 뒤 변환합니다. 오래된 캐시가 남아 있으면 형상 불일치 오류가 발생합니다.

**2단계 — 동적 범위 양자화 (가중치 INT8, 활성화 float32):**

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

캘리브레이션 데이터 없이도 full INT8 대비 동등한 크기 감소를 달성합니다. full INT8(활성화까지 양자화)은 `onnx2tf` 변환 모델에서 출력 스케일 불일치를 일으키므로 사용하지 않습니다.

**수정된 버그 (NCHW → NHWC):**

PyTorch는 NCHW 포맷이지만 onnx2tf 변환 후 TF 모델은 NHWC를 기대합니다.
`np.transpose` 제거 후 `np.expand_dims`로 올바른 shape 생성:

```python
# 수정 전 (버그)
img_array = np.transpose(img_array, (2, 0, 1))  # (3,224,224) NCHW
img_array = np.expand_dims(img_array, axis=0)   # (1,3,224,224) 잘못됨

# 수정 후
img_array = np.expand_dims(img_array, axis=0)   # (1,224,224,3) NHWC 올바름
```

```bash
python convert_to_tflite.py
```

출력: `./weights/auralens_model_int8.tflite`

---

## 6단계: 모델 평가

**스크립트:** `evaluate_models.py`

ONNX 모델과 TFLite INT8 모델의 정확도를 전체 검증 데이터셋 기준으로 비교합니다. 양자화로 인한 정확도 손실이 허용 범위인지 확인한 뒤 배포 여부를 결정합니다.

**전처리:** 두 모델 모두 동일한 전처리를 사용합니다.

```python
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

img = Image.open(path).convert('RGB').resize((224, 224))
arr = (arr - MEAN) / STD          # NHWC (1, 224, 224, 3)
```

ONNX 추론 시에는 `np.transpose`로 NCHW로 변환합니다. TFLite는 NHWC 그대로 입력합니다.

**출력 — 리포트:**

- 전체 정확도
- 클래스별 정확도 (person / food / scenery)
- Confusion Matrix

**양자화 손실 판정:**

| 손실 | 판정 | 조치 |
|------|------|------|
| ≤ 2%p | 양호 — 배포 가능 | — |
| 2~5%p | 허용, 개선 여지 있음 | 실기기 테스트 후 결정 |
| > 5%p | 재변환 필요 | representative 샘플 수 확대 또는 2단계 학습 전략 적용 |

```bash
python evaluate_models.py                  # 전체 검증 데이터
python evaluate_models.py --limit 300      # 클래스당 최대 300장
python evaluate_models.py --onnx_only      # ONNX만 평가 (TFLite 없을 때)
```

---

## 7단계: Flutter 앱 통합

### Flutter 앱 전체 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                     main.dart                        │
│  ModelGate.probe()          ←── 앱 시작 최초 1회     │
│  InferenceService.initialize()                       │
│  Provider 트리 구성                                   │
└──────────────────────┬──────────────────────────────┘
                       │ Provider
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   CameraService  InferenceService  CameraViewModel
   (카메라 HW)    (ONNX 추론)       (비즈니스 로직)
          │                              │
          │ 카메라 프레임 (YUV420)        │ notifyListeners()
          └──────────────────────────────┘
                       │
               CameraScreen (UI)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   CameraPreview              CompositionOverlayPainter
   (라이브 뷰)                 (그리드 + AI 오버레이)
```

### 제어 흐름 — ONNX 결과가 ML Kit 활성화를 결정

두 추론 엔진은 같은 카메라 스트림 콜백 안에서 **순차적으로** 실행됩니다. ONNX 장면 분류 결과가 ML Kit 실행 여부를 게이팅합니다.

```
CameraImage 수신 (0.5초 쿨다운)
        │
        ▼
  ONNX 장면 분류
  (ImageConverter → InferenceService)
        │
        ▼
  7프레임 다수결 → stableScene
  (항상 person / food / scenery 중 하나 — 모델 출력이 3개이므로 unknown 불가)
        │
        ├─ person or food ──► ML Kit 객체 감지 실행
        │                          │ (YUV_420_888 → NV21 변환)
        │                          ▼
        │                    scaleRect → CompositionService
        │                          │
        │                    detectedBoundingBox
        │                    compositionTarget  (수동 포커스 중에는 덮어쓰지 않음)
        │                    isCompositionCorrect
        │                          │
        │                          ▼
        │                    자동 AF/AE (수동 포커스 비활성 시)
        │                    └─ 2초 경과 또는 피사체 이동 15% 초과 시
        │                       CameraService.setFocusAndExposure()
        │
        └─ scenery ──► ML Kit 스킵
                       detectedBoundingBox = null

[별도 경로] 화면 탭 (GestureDetector.onTapUp)
        │
        ▼
  CameraViewModel.onScreenTap(localPosition, widgetSize)
        ├─ compositionTarget = findClosestPowerPoint(탭 위치)
        ├─ _isManualFocusActive = true
        └─ CameraService.setFocusAndExposure(정규화 좌표)

[별도 경로] 가속도계 (userAccelerometerEventStream)
        │  벡터 크기 > 3.5 m/s²  &&  수동 포커스 활성
        ▼
  _resetManualFocus()
        └─ CameraService.resetAutoFocusExposure()
```

`SceneCategory.unknown`은 `_currentScene`의 초기값 및 AI 비활성 시 리셋 전용 상태입니다. `stableScene`(모델 argmax 결과)에는 절대 도달하지 않으므로 위 분기에 포함되지 않습니다.

이 설계로 scenery 씬에서 ML Kit 연산 비용이 발생하지 않으며, 두 모델이 같은 프레임을 처리할 때 생기는 레이스 컨디션도 없습니다 (`_isDetecting` 플래그로 동시 실행 방지).

---

### 7-0. ModelGate (`model_gate.dart`)

앱 시작 시 모델 파일 존재 여부를 단 한 번 확인하고, 이후 `InferenceService`가 이 상태를 참조해 초기화를 결정합니다.

```
ModelGate.probe()                          ← main() 에서 최초 1회 호출
  │
  ├─ rootBundle.load('assets/models/auralens_model.onnx') 성공
  │    → _isReady = true
  │    → 로그: 🔓 AI 모드
  │
  └─ FlutterError (파일 없음)
       → _isReady = false
       → 로그: 🔒 카메라 전용 모드

ModelGate.isReady                          ← InferenceService 에서 참조
```

모델 파일 부재를 예외(Exception)가 아닌 명시적 상태로 다룹니다. 파일을 추가하면 코드 수정 없이 다음 실행부터 게이트가 열립니다.

**카메라 전용 모드 (모델 파일 없음):**

```
ModelGate.probe() → _isReady = false
  ↓
InferenceService.initialize() → 즉시 반환 (_isInitialized = false)
  ↓
classifyScene() → SceneCategory.unknown 즉시 반환
  ↓
CompositionOverlayPainter: unknown 분기 = 아무것도 그리지 않음
  └─ 그리드·셔터·갤러리 등 카메라 기능은 모두 정상 동작
```

---

### 7-1. 장면 분류 (`inference_service.dart`)

`auralens_model.onnx`를 ONNX Runtime (`ort`)으로 로드하여 카메라 프레임을 추론합니다.

```
initialize()
  ├─ ModelGate.isReady == false → 즉시 반환
  └─ ModelGate.isReady == true
       └─ OrtSession.fromBuffer() 로 모델 로드
            ├─ 성공 → _isInitialized = true
            └─ 실패 → _isInitialized = false

classifyScene(Float32List inputData) → Future<SceneCategory>
  ├─ _isInitialized == false → SceneCategory.unknown 즉시 반환
  └─ 추론 실행
       입력: [1, 3, 224, 224] NCHW Float32List
       출력: [[p_person, p_food, p_scenery]]
       argmax → SceneCategory
```

출력 인덱스: 0=person, 1=food, 2=scenery

리소스 관리: `inputTensor`, `outputs`, `runOptions`를 `finally` 블록에서 `.release()` 합니다.

**7프레임 다수결 안정화:**

```dart
_sceneHistory.add(rawResult);
if (_sceneHistory.length > 7) _sceneHistory.removeAt(0);
final stableScene = voteCount.entries.reduce((a, b) => a.value > b.value ? a : b).key;
```

---

### 7-2. 카메라 서비스 (`camera_service.dart`)

Flutter `camera` 패키지를 감싼 ChangeNotifier 서비스. 카메라 하드웨어와의 접점입니다.

| 메서드 | 설명 |
|--------|------|
| `initializeCamera()` | 기기에서 카메라 목록을 조회하고 YUV420 포맷으로 컨트롤러를 초기화합니다 |
| `startImageStream(callback)` | 매 프레임을 콜백으로 흘려보내는 스트림을 시작합니다 |
| `stopImageStream()` | 스트림을 정지합니다 (갤러리 진입 시 호출) |
| `takePicture()` | 정사진을 촬영해 `XFile`로 반환합니다 |
| `switchCamera()` | 전/후면 카메라를 전환합니다 |
| `updateCameraResolution()` | 해상도 변경 시 컨트롤러를 재초기화합니다 |
| `setFocusAndExposure(Offset)` | 정규화 좌표(0~1)로 포커스 모드를 locked로 전환하고 포커스·노출 포인트를 설정합니다 |
| `resetAutoFocusExposure()` | 포커스·노출 모드를 auto로 복원하고 포인트를 null로 초기화합니다 |

`imageFormatGroup: ImageFormatGroup.yuv420`으로 고정되어 있어, `ImageConverter`가 항상 YUV420 입력을 받는다고 가정할 수 있습니다.

---

### 7-3. 객체 감지 (`ML Kit`)

`google_mlkit_object_detection`의 `DetectionMode.stream`으로 실시간 바운딩 박스를 추출합니다.
ONNX 결과가 `person` 또는 `food`일 때만 호출됩니다 (위 제어 흐름 참고).

**YUV_420_888 → NV21 변환 후 InputImage 전달:**

`camera_android_camerax`(CameraX 백엔드)가 전달하는 이미지는 내부적으로 `YUV_420_888`입니다.
`InputImageFormat.yuv420`은 iOS 전용 포맷이므로 Android에서는 `PlatformException(InputImageConverterError)` 오류가 발생합니다.
각 Y/U/V 평면을 행 단위로 복사해 NV21(= Y 평면 + VU 인터리브)로 수동 변환한 뒤 `InputImageFormat.nv21`로 전달해야 합니다.

```dart
// Y 평면 행 단위 복사 (패딩 제거)
for (int row = 0; row < h; row++) {
  nv21.setRange(row * w, row * w + w, yPlane.bytes, row * yRowStride);
}
// VU 인터리브 (NV21: V 먼저, U 다음)
for (int row = 0; row < h ~/ 2; row++) {
  for (int col = 0; col < w ~/ 2; col++) {
    final int srcIdx = row * uvRowStride + col * uvPixStride;
    nv21[uvDst++] = vPlane.bytes[srcIdx];
    nv21[uvDst++] = uPlane.bytes[srcIdx];
  }
}

InputImage.fromBytes(
  bytes: nv21,
  metadata: InputImageMetadata(
    size: Size(w.toDouble(), h.toDouble()),
    rotation: InputImageRotation.rotation90deg,
    format: InputImageFormat.nv21,   // ← Android CameraX 전용
    bytesPerRow: w,
  ),
);
```

**좌표 변환 (rotation90deg 적용):**

```dart
final rotatedSize = Size(cameraImage.height.toDouble(), cameraImage.width.toDouble());
_detectedBoundingBox = scaleRect(
  rect: objects.first.boundingBox,
  imageSize: rotatedSize,  // 회전 후 가로/세로 교환
  widgetSize: _screenSize,
);
```

---

### 7-4. 구도 서비스 (`composition_service.dart`)

AI 추론 결과(좌표)를 구도 계산 로직으로 변환하는 서비스입니다.

**`findClosestPowerPoint(detectionBox, screenSize)`**

바운딩 박스 중심에서 3분할 4개 교차점(파워 포인트) 중 가장 가까운 점을 반환합니다. 허용 오차 = 화면 너비의 5%.

**`calculateTriangularCompositionTargets(detections, ...)`**

감지된 객체 2개 이상을 받아 삼각형 구도의 세 꼭짓점 `TriangularComposition`을 반환합니다. 가장 큰 두 객체의 중심 + 화면 상단 1/4 지점으로 구성됩니다(향후 고도화 예정).

> **현재 상태:** 메서드들은 구현 완료되었으나 `CameraViewModel`에 아직 연결되지 않았습니다. `CompositionOverlayPainter`가 직접 장면 카테고리로 분기해 고정 가이드를 그리는 방식으로 동작 중입니다.

---

### 7-5. 비즈니스 로직 (`camera_view_model.dart`)

카메라 화면의 비즈니스 로직 전담 ChangeNotifier. `CameraService`와 `InferenceService`를 조율합니다.

**AI 추론 루프 (`_startModelInference`)**

```
CameraService.startImageStream()
  │ 매 프레임 (YUV420)
  ▼
500ms 쿨다운 체크 (너무 빠른 프레임 스킵)
  │
  ▼
compute(ImageConverter.convertCameraImageToModelInput, frame)
  │ 별도 isolate (UI 블로킹 없음)
  ▼
InferenceService.classifyScene(Float32List)
  │
  ▼
_sceneHistory 에 결과 추가 (최대 7개 보관)
  │
  ▼
다수결 투표 → stableScene
  │
  ▼
_currentScene 변경 시 notifyListeners()
```

**주요 상태값**

| 필드 | 타입 | 역할 |
|------|------|------|
| `_currentScene` | `SceneCategory` | 안정화된 현재 장면 |
| `_isAiAssistEnabled` | `bool` | AI 어시스트 토글 |
| `_isGridEnabled` | `bool` | 그리드 표시 토글 |
| `_currentRatio` | `CameraRatio` | 화면 비율 |
| `_sessionPhotos` | `List<XFile>` | 세션 내 촬영 사진 목록 |
| `_detectedBoundingBox` | `Rect?` | ML Kit 바운딩 박스 (위젯 좌표) |
| `_compositionTarget` | `Offset?` | 가장 가까운 파워포인트 |
| `_isCompositionCorrect` | `bool` | 구도 일치 여부 |
| `_isManualFocusActive` | `bool` | 수동 포커스 활성 여부 (탭 시 true) |
| `_focusTapPoint` | `Offset?` | 탭한 위치 (포커스 인디케이터용) |

**자동 포커스 (`_runObjectDetection` 내부):**

ML Kit가 피사체를 감지했을 때, 수동 포커스가 비활성 상태이면 피사체 중심으로 자동 포커스·노출을 설정합니다.
직전 포커스 설정 시각으로부터 2초가 지났거나, 피사체가 화면 너비의 15% 이상 이동한 경우에만 재설정합니다 (과도한 AF 발동 방지).

```dart
if (!_isManualFocusActive) {
  final movedFar = _lastAutoFocusPoint != null &&
      (_lastAutoFocusPoint! - subjectCenter).distance > _screenSize.width * 0.15;
  if (elapsed >= 2s || movedFar) {
    _cameraService.setFocusAndExposure(normalizedCenter);
  }
}
```

**수동 포커스 · 구도 업데이트 (`onScreenTap`):**

화면 탭 시 `_isManualFocusActive = true`로 전환되고, 탭 위치를 `CompositionService.findClosestPowerPoint`에 전달해 구도 목표를 즉시 업데이트합니다. 동시에 정규화 좌표로 `CameraService.setFocusAndExposure()`를 호출해 카메라 포커스·노출을 잠급니다.

**움직임 기반 수동 포커스 해제 (`_accelSub`):**

`userAccelerometerEventStream`(중력 제거된 선형 가속도)을 구독합니다. 벡터 크기가 3.5 m/s² 초과 시 `_resetManualFocus()`를 호출해 auto 모드로 복원합니다. 카메라 전환(`switchCamera`) 시에도 자동 초기화됩니다.

**수동 포커스 상태에서의 ML Kit 동작:**

- `_isManualFocusActive == true`이면 `_compositionTarget`을 ML Kit 결과로 덮어쓰지 않습니다.
- 피사체가 없어 `objects.isEmpty`여도 수동 포커스 구도 목표는 유지됩니다.

**AI 일시정지/재개:** 갤러리 진입 시 `pauseInference()` → 이미지 스트림 정지, 복귀 시 `resumeInference()` → 스트림 재시작.

---

### 7-6. 오버레이 렌더링 (`composition_overlay_painter.dart`)

`CustomPainter` 구현체. 카메라 프리뷰 위에 투명하게 덮이는 구도 안내 오버레이를 그립니다.

씬별 분기 없이 황금비(φ = 1.618…) 기반의 **통합 오버레이** 하나로 모든 장면을 처리합니다.

**렌더링 분기**

```
isGridEnabled == true
  └─ _drawGrid() : 흰/검 2중 3분할 격자선

isAiAssistEnabled == true  &&  currentScene != unknown
  │
  ├─ mainPoint = compositionTarget ?? _defaultMainPoint(size)
  │   (ML Kit 결과가 없으면 씬별 기본 파워포인트 사용)
  │
  ├─ _drawLeadingLines()
  │   화면 하단 양 모서리 → mainPoint 수렴 점선 + MaskFilter.blur 글로우
  │
  ├─ _drawMainCircle()
  │   반지름 = width / φ³ ≈ width × 0.236 (피보나치 반지름)
  │   ├─ 구도 불일치 → 점선 원 + 약한 글로우 (pulse 진폭 소)
  │   └─ 구도 일치   → 실선 원 + 강한 글로우 (pulse 진폭 대, AnimationController)
  │
  └─ detectedBoundingBox != null  &&  !isCompositionCorrect
      └─ _drawSubjectMarker() : 현재 피사체 위치 십자(+) 마커
```

**씬별 색상 · 기본 구도 목표**

| SceneCategory | 색상 | 기본 mainPoint |
|---|---|---|
| `person` | light blue `#80D8FF` | 좌상단 파워포인트 (W/3, H/3) |
| `food` | amber `#FFD740` | 화면 중앙 (W/2, H/2) |
| `scenery` | white | 화면 중앙 (W/2, H/2) |
| `unknown` | — | 오버레이 비표시 |

**글로우 구현:** `MaskFilter.blur(BlurStyle.normal, σ)`로 실제 블러를 적용합니다. 구도 일치 시 `AnimationController`의 pulse 값(0→1→0, 1.5초 반복)이 불투명도·블러 반경을 변조합니다.

`shouldRepaint()`는 `currentScene`, `isGridEnabled`, `isAiAssistEnabled`, `widgetSize`, `detectedBoundingBox`, `compositionTarget`, `isCompositionCorrect` 변경 시 재드로우합니다.

---

### End-to-End 흐름 (AI 모드)

```
① assets/models/auralens_model.onnx 배치
         │
         ▼
② 앱 시작 → ModelGate.probe()
         │  파일 확인 성공 → _isReady = true / 로그: 🔓 AI 모드
         ▼
③ InferenceService.initialize()
         │  OrtSession 로드 → _isInitialized = true
         ▼
④ CameraService.initializeCamera()
         │  YUV420 스트림 준비
         ▼
⑤ CameraViewModel._startModelInference()
         │  [500ms마다]
         ▼
⑥ compute(ImageConverter.convertCameraImageToModelInput, frame)
         │  YUV420 → RGB → 224×224 → NCHW Float32List
         │  (별도 isolate, UI 블로킹 없음)
         ▼
⑦ InferenceService.classifyScene(Float32List)
         │  ONNX 추론: [1,3,224,224] → [[p0, p1, p2]]
         │  argmax → SceneCategory
         ▼
⑧ _sceneHistory 누적 (최대 7프레임 다수결)
         │  흔들림/오탐 방지용 안정화
         ▼
⑨ _currentScene 업데이트 → notifyListeners()
         ▼
⑩ CompositionOverlayPainter.paint()
         ├─ 3분할 그리드 (isGridEnabled)
         └─ 피보나치 오버레이 (isAiAssistEnabled)
              ├─ 하단 양 모서리 → mainPoint 수렴 점선 + 글로우
              ├─ mainPoint 원 (반지름 = width/φ³) + 글로우
              │   ├─ 구도 불일치 → 점선
              │   └─ 구도 일치  → 실선 + pulse 애니메이션
              └─ 피사체 마커 (구도 불일치 시 십자)
```

---

## 파일 구조

```
Auralens_ai_project/
  data/                        ← 커스텀 레퍼런스 이미지
  raw_datasets/                ← Kaggle 원본 데이터
  dataset/                     ← 학습/검증 분리 완료
    train/ val/
  weights/
    best_auralens_model.pth    ← PyTorch 체크포인트
    auralens_model.onnx        ← ONNX 배포 모델
    auralens_model_int8.tflite ← TFLite 모바일 모델
  process_custom_data.py
  prepare_dataset.py
  train.py
  export_tflite.py
  convert_to_tflite.py
  evaluate_models.py

AuraLens/lib/src/
  services/
    model_gate.dart             ← 모델 파일 존재 확인 게이트
    inference_service.dart      ← ONNX Runtime 추론
    camera_service.dart         ← 카메라 스트림 관리
    composition_service.dart    ← 구도 파워포인트 계산
  screens/camera/
    camera_view_model.dart      ← ML Kit + 추론 루프
    camera_screen.dart          ← UI 진입점
    gallery_screen.dart         ← 세션 사진 뷰어 (핀치줌·공유·삭제)
    widgets/
      composition_overlay_painter.dart  ← 오버레이 렌더링
  screens/settings/
    settings_screen.dart        ← 해상도·그리드·AI 토글 설정
  utils/
    image_converter.dart        ← CameraImage → 모델 입력
    coordinate_scaler.dart      ← 이미지→위젯 좌표 변환
  models/
    camera_settings.dart        ← SceneCategory 등 열거형
```

---

## 패키지 의존성

| 패키지 | 역할 |
|--------|------|
| `camera` | 카메라 프리뷰 및 YUV420 스트림 |
| `onnxruntime` | ONNX 모델 추론 엔진 |
| `google_mlkit_object_detection` | 객체 감지 (바운딩 박스) |
| `image` | YUV→RGB 변환, 리사이징 |
| `provider` | 상태 관리 (DI 컨테이너) |
| `sensors_plus` | 가속도계 (아이콘 회전) + 선형 가속도 (수동 포커스 움직임 감지) |
| `share_plus` | 사진 공유 |
| `gallery_saver_plus` | 갤러리 저장 |
| `permission_handler` | 카메라·저장소 권한 |

---

## 모델 연결 체크리스트

### 1단계 — PyTorch 모델 Export 확인

```python
torch.onnx.export(
    model,
    dummy_input,                      # shape: (1, 3, 224, 224)
    "auralens_model.onnx",
    input_names=["input"],            # ← 반드시 'input' 으로 지정
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch_size"}},
)
```

- [ ] `input_names=['input']` 지정 확인 (`inference_service.dart`의 `{'input': inputTensor}` 키와 일치)
- [ ] 입력 shape `(1, 3, 224, 224)` NCHW 확인
- [ ] 출력 shape `(1, 3)` — 3개 클래스 확률값(softmax 또는 logits) 확인
- [ ] `opset_version=11` 이상 확인

### 2단계 — 클래스 순서 확인

앱 내 클래스 인덱스 매핑 (`inference_service.dart`):

| 인덱스 | SceneCategory |
|--------|---------------|
| 0 | `person` |
| 1 | `food` |
| 2 | `scenery` |

- [ ] 학습 시 클래스 순서가 위 표와 일치하는지 확인 (`_AppOrderWrapper` 적용 확인)
- [ ] 순서가 다르면 `inference_service.dart`의 `switch(maxIndex)` 블록만 수정

### 3단계 — 파일 배치

```
AuraLens/
└── assets/
    └── models/
        └── auralens_model.onnx   ← 여기에 복사
```

- [ ] `assets/models/auralens_model.onnx` 파일 복사 완료
- [ ] `pubspec.yaml`의 `assets/models/` 디렉터리 등록 확인

### 4단계 — 빌드 및 로그 확인

- [ ] `flutter pub get` 실행
- [ ] 앱 실행 후 디버그 콘솔에서 아래 로그 확인:
  ```
  🔓 ModelGate: assets/models/auralens_model.onnx 확인됨 → AI 모드
  ✅ ONNX 모델 로드 성공
  ```
- [ ] 카메라 화면에서 AI 어시스트 아이콘(insights) 활성화 후 오버레이 등장 확인
- [ ] 음식 / 인물 / 풍경 장면별 오버레이 색상 및 형태 확인

---

## 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| `🔒 카메라 전용 모드` 로그 | 파일 경로 오류 또는 파일 미복사 | `assets/models/auralens_model.onnx` 위치 재확인 |
| `❌ ONNX 모델 로드 실패` 로그 | opset 불일치 또는 파일 손상 | `opset_version` 재확인 후 재export |
| 오버레이가 항상 나타나지 않음 | 클래스 순서 불일치로 `unknown` 반환 | 학습 레이블 순서와 `switch` 블록 비교 |
| 추론 결과가 불안정함 | 정규화 값 불일치 | ImageNet mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]` 확인 |
| TFLite 정확도 손실 > 5%p | representative 샘플 부족 | `evaluate_models.py` 실행 후 `convert_to_tflite.py` 샘플 수 확대 |
| `onnx2tf` 변환 형상 오류 | 이전 SavedModel 캐시 잔존 | `./weights/tf_saved_model/` 폴더 수동 삭제 후 재실행 |
| `PlatformException(InputImageConverterError, ImageFormat is not supported.)` | `InputImageFormat.yuv420`은 iOS 전용 포맷 — CameraX Android에서 지원 안됨 | `_buildInputImage`에서 YUV_420_888 → NV21 수동 변환 후 `InputImageFormat.nv21` 사용 |
| 탭해도 포커스가 바뀌지 않음 | 기기가 `focusPointSupported` / `exposurePointSupported` == false | `CameraController.value`로 지원 여부 확인; 미지원 기기는 AF가 고정됨 (정상) |
| 움직여도 수동 포커스가 해제되지 않음 | `userAccelerometerEventStream` 미지원 기기 또는 threshold 과도 | `_moveThreshold` 값(3.5 m/s²) 낮추거나 탭 재시도로 수동 해제 |
| ONNX IR 버전 오류 (`max supported IR version: 9`) | PyTorch 2.x는 IR version 10으로 내보냄 | `export_tflite.py`의 IR 다운그레이드 블록 실행 확인; 또는 `onnx.load` 후 `model_proto.ir_version = 9` 패치 |
