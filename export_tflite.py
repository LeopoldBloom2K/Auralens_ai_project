import torch
import torch.nn as nn
from torchvision import models
import os

def export_to_onnx():
    print("🔄 PyTorch 모델을 ONNX로 변환 시작...")
    
    # 1. 모델 구조 생성
    model = models.mobilenet_v3_large()
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, 3) # 클래스 개수 3개 (food, person, scenery)
    
    # 2. 학습된 가중치 불러오기
    weight_path = './weights/best_auralens_model.pth'
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    # 3. ONNX 변환을 위해 모델의 자동 풀링을 고정된 크기로 변경
    model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    
    model.eval() 

    # 배치 사이즈를 1로 고정
    dummy_input = torch.randn(1, 3, 224, 224)

    onnx_path = "./weights/auralens_model.onnx"
    
    torch.onnx.export(
        model,                      
        dummy_input,                
        onnx_path,                  
        export_params=True,         
        opset_version=14,           
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output']
    )
    
    print(f"✅ ONNX 변환 완료! 파일 위치: {onnx_path}")

if __name__ == '__main__':
    export_to_onnx()
    