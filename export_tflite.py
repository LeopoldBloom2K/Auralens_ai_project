import torch
import torch.nn as nn
from torchvision import models
import os
import onnx

NUM_CLASSES = 3

# ImageFolder는 알파벳 순으로 클래스를 정렬합니다.
# 학습 시 자동 부여 순서: food=0, person=1, scenery=2
# Auralens 앱이 기대하는 순서: person=0, food=1, scenery=2
# → 아래 래퍼가 출력 텐서를 앱 기대 순서로 재배열합니다.
_PERM = [1, 0, 2]  # [person, food, scenery]


class _AppOrderWrapper(nn.Module):
    """출력 클래스 순서를 앱 기대 순서(person=0, food=1, scenery=2)로 재배열."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)
        return torch.stack([out[:, 1], out[:, 0], out[:, 2]], dim=1)


def export_to_onnx():
    print("🔄 PyTorch 모델을 ONNX로 변환 시작...")

    # 1. 모델 구조 생성 (train.py 와 완전히 동일하게)
    model = models.mobilenet_v3_large()
    num_features = model.classifier[3].in_features  # 1280
    model.classifier[3] = nn.Linear(num_features, NUM_CLASSES)

    # 2. 학습된 가중치 불러오기
    weight_path = './weights/best_auralens_model.pth'
    if not os.path.exists(weight_path):
        print(f"❌ 가중치 파일을 찾을 수 없습니다: {weight_path}")
        print("   train.py를 먼저 실행해주세요!")
        return
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    # 3. ONNX 변환을 위해 적응형 풀링을 고정 크기로 변경 (224×224 입력 기준 7×7 특징 맵)
    model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

    model.eval()

    # 4. 출력 순서를 앱 기대 순서로 재배열하는 래퍼 적용
    export_model = _AppOrderWrapper(model)
    export_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    os.makedirs('./weights', exist_ok=True)
    onnx_path = './weights/auralens_model.onnx'

    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}},
    )

    # onnxruntime Flutter 패키지(ORT 1.17.x)는 IR version 9까지 지원.
    # 최신 PyTorch/onnx는 IR version 10으로 내보내므로 9로 낮춥니다.
    model_proto = onnx.load(onnx_path)
    if model_proto.ir_version > 9:
        print(f"   IR 버전 다운그레이드: {model_proto.ir_version} → 9")
        model_proto.ir_version = 9
        onnx.save(model_proto, onnx_path)

    print(f"✅ ONNX 변환 완료! 파일 위치: {onnx_path}")
    print("   출력 클래스 순서: person=0, food=1, scenery=2  (inference_service.dart 와 일치)")


if __name__ == '__main__':
    export_to_onnx()
