import os
import random
import shutil
from pathlib import Path

def prepare_dataset():
    print("🚀 AuraLens 데이터셋 자동 분류를 시작합니다...")

    # 1. 뼈대가 될 최종 데이터셋 폴더 구조 만들기
    base_dir = Path('./dataset')
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    classes = ['person', 'food', 'scenery', 'unknown']

    for dir_path in [train_dir, val_dir]:
        for cls in classes:
            (dir_path / cls).mkdir(parents=True, exist_ok=True)

    # 2. 알려준 정확한 디렉토리 구조 반영 (rglob이 하위 폴더를 자동 탐색함)
    raw_data_paths = {
        'person': [
            Path('./raw_datasets/human-images-dataset-men-and-women/gender_dataset/men'),
            Path('./raw_datasets/human-images-dataset-men-and-women/gender_dataset/women')
        ],
        'food': [
            Path('./raw_datasets/food11-image-dataset/training'),
            Path('./raw_datasets/food11-image-dataset/validation'),
            Path('./raw_datasets/food11-image-dataset/evaluation')
        ],
        'scenery': [
            Path('./raw_datasets/intel-image-classification/seg_train/seg_train'),
            Path('./raw_datasets/intel-image-classification/seg_test/seg_test')
        ],
        'unknown': [
            Path('./raw_datasets/coco_val2017/val2017')
        ]
    }

    # 각 클래스당 최대 몇 장의 사진을 사용할 것인지 (데이터 불균형 방지용)
    max_images_per_class = 5000 
    split_ratio = 0.8 # 학습(Train) 80%, 검증(Val) 20%

    for cls in classes:
        print(f"\n📂 '{cls}' 카테고리 이미지 수집 중...")
        all_images = []
        
        # 지정된 경로들에서 모든 jpg 이미지 싹쓸이 수집
        for folder in raw_data_paths[cls]:
            if not folder.exists():
                print(f"  ⚠️ 경고: 폴더를 찾을 수 없습니다 -> {folder}")
                continue
            
            # 하위 폴더(Bread, buildings 등)까지 싹 뒤져서 확장자 파일 찾기
            images = list(folder.rglob('*.jpg')) + list(folder.rglob('*.jpeg')) + list(folder.rglob('*.png'))
            all_images.extend(images)

        # 혹시 모를 중복 경로 제거
        all_images = list(set(all_images))

        # 사진 무작위로 섞기 및 개수 제한
        random.shuffle(all_images)
        if len(all_images) > max_images_per_class:
            all_images = all_images[:max_images_per_class]

        total_imgs = len(all_images)
        if total_imgs == 0:
            print(f"  ❌ 에러: '{cls}' 카테고리 이미지가 0장입니다. 압축 해제 폴더명을 확인하세요.")
            continue

        train_count = int(total_imgs * split_ratio)
        train_images = all_images[:train_count]
        val_images = all_images[train_count:]

        print(f"  -> 총 {total_imgs}장 발견! (학습용: {len(train_images)}장, 검증용: {len(val_images)}장)")
        print(f"  -> 사진 복사 및 이름 변경 중... 잠시만 기다려주세요.")

        # 3. 파일 복사 및 이름 중복 방지를 위한 넘버링
        for i, img_path in enumerate(train_images):
            new_name = f"{cls}_train_{i}.jpg"
            shutil.copy(img_path, train_dir / cls / new_name)
            
        for i, img_path in enumerate(val_images):
            new_name = f"{cls}_val_{i}.jpg"
            shutil.copy(img_path, val_dir / cls / new_name)

    print("\n✅ 모든 데이터셋 세팅이 완벽하게 끝났습니다! AI 학습 준비 완료!")

if __name__ == '__main__':
    prepare_dataset()