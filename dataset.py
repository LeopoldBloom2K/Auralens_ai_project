import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    # 1. 학습용 데이터 전처리 (Data Augmentation 포함)
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),                                  # 일단 살짝 크게 리사이징
        transforms.RandomCrop(224),                                     # 224x224 크기로 무작위 자르기 (다양성 확보)
        transforms.RandomHorizontalFlip(),                              # 50% 확률로 좌우 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2),           # 밝기, 대비 무작위 변경
        transforms.ToTensor(),                                          # 이미지를 PyTorch Tensor로 변환 (0~1 사이 값)
        transforms.Normalize(                                           # MobileNetV3 사전학습 모델에 맞춘 정규화
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. 검증용 데이터 전처리 (평가만 하므로 증강 없이)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 3. 폴더에서 이미지 불러오기 (자동 라벨링)
    # 💡 꿀팁: ImageFolder는 폴더 이름을 알파벳 순으로 정렬해 클래스 인덱스를 자동 부여합니다. (food=0, person=1, scenery=2)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

    # 4. DataLoader 생성 (GPU로 데이터를 빠르게 쏴주는 역할)
    # pin_memory=True, num_workers를 설정하면 로컬 GPU 자원을 최대한 활용
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes

# 테스트용 코드
if __name__ == "__main__":
    # 🚀 수정: 데이터 폴더 경로를 프로젝트 구조에 맞게 './dataset'으로 변경
    try:
        train_loader, val_loader, classes = get_dataloaders(data_dir='./dataset', batch_size=16)
        
        # 자동으로 3개의 클래스(['food', 'person', 'scenery'])가 잡히는지 확인합니다.
        print(f"✅ 자동으로 찾아낸 클래스 목록: {classes}") 
        
        # 첫 번째 배치(Batch) 확인
        images, labels = next(iter(train_loader))
        print(f"이미지 텐서 형태: {images.shape}") # 예상: [16, 3, 224, 224]
        print(f"라벨 텐서 형태: {labels.shape}")   # 예상: [16]
        
    except FileNotFoundError:
        print("❌ dataset/ 폴더 구조를 먼저 만들어주세요! (prepare_dataset.py를 실행해야 합니다)")