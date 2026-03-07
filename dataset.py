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
    # 데이터 폴더 경로 지정 (실제 폴더 구조에 맞게 수정)
    # 미리 data/train, data/val 폴더를 만들고 임의의 이미지 넣고 실행
    try:
        train_loader, val_loader, classes = get_dataloaders(data_dir='./data', batch_size=16)
        print(f"찾아낸 클래스 목록: {classes}")
        
        # 첫 번째 배치(Batch) 확인
        images, labels = next(iter(train_loader))
        print(f"이미지 텐서 형태: {images.shape}") # 예상: [16, 3, 224, 224]
        print(f"라벨 텐서 형태: {labels.shape}")   # 예상: [16]
    except FileNotFoundError:
        print("data/ 폴더 구조를 먼저 만들어주세요!")