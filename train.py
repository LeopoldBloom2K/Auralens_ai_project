import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time

def train_model():
    # 1. 사용 가능한 GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 시작! 사용 장치: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # 2. 데이터 증강(Augmentation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),           # 색감, 밝기 무작위 변형
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. 데이터셋 및 데이터로더 설정
    data_dir = './dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    
    # batch_size는 VRAM에 무리가 가지 않도록 32로 설정 (여유가 있다면 64로 올려도 됨)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    print(f"📸 인식할 클래스 목록: {class_names}")
    print(f"📊 데이터 수: 학습용 {dataset_sizes['train']}장 / 검증용 {dataset_sizes['val']}장\n")

    # 4. MobileNetV3 모델 불러오기 및 수정 (전이 학습)
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    
    # 마지막 분류기 부분을 우리가 가진 클래스 개수(3개)에 맞게 뜯어고침
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(class_names))
    model = model.to(device)

    # 5. 오차 함수와 최적화 도구 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 6. 본격적인 학습 루프
    num_epochs = 10  # 일단 10번 반복 (결과를 보고 나중에 늘려도 됨)
    best_acc = 0.0

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 15)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 학습 모드
            else:
                model.eval()   # 검증(평가) 모드

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 배치(32장) 단위로 가져와서 학습
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 학습 모드일 때만 기울기(Gradient) 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 모드일 때만 역전파 및 가중치 업데이트
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize():>5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

            # 검증 정확도가 역대 최고 기록을 깼을 때만 모델을 덮어쓰기 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs('./weights', exist_ok=True)
                torch.save(model.state_dict(), './weights/best_auralens_model.pth')
                print("  🌟 최고 성능 갱신! (best_auralens_model.pth 저장 완료)")

        print() # 에포크 간 줄바꿈

    time_elapsed = time.time() - start_time
    print(f'🎉 학습 완전 종료! 소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')
    print(f'🏆 최종 최고 검증 정확도(Best Val Acc): {best_acc:.4f}')

if __name__ == '__main__':
    train_model()