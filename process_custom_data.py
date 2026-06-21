# -*- coding: utf-8 -*-
import shutil
from pathlib import Path

# data/ 폴더 그룹별 클래스 매핑 (파일명 타임스탬프 기준)
# 000014850 -> person  (인물+풍경, person 데이터 부족으로 포함)
# 000030313 -> scenery (실내.도시 공간)
# 000057355 -> food    (음식 사진 위주)
# 000043833 -> food    (음료.음식)
# 000337932 -> scenery (자연.건축 경관)
GROUP_CLASS_MAP = {
    'KakaoTalk_20260621_000014850': 'person',
    'KakaoTalk_20260621_000030313': 'scenery',
    'KakaoTalk_20260621_000057355': 'food',
    'KakaoTalk_20260621_000043833': 'food',
    'KakaoTalk_20260621_000337932': 'scenery',
}

# prepare_dataset.py 의 rglob 탐색 경로 하위에 custom/ 서브폴더를 생성합니다.
DST_DIRS = {
    'person':  Path('./raw_datasets/human-images-dataset-men-and-women/gender_dataset/men/custom'),
    'food':    Path('./raw_datasets/food11-image-dataset/training/custom'),
    'scenery': Path('./raw_datasets/intel-image-classification/seg_train/seg_train/custom'),
}


def process_custom_data():
    src_dir = Path('./data')

    if not src_dir.exists():
        print("ERROR: ./data 폴더가 없습니다.")
        return

    for dst in DST_DIRS.values():
        dst.mkdir(parents=True, exist_ok=True)

    copied = {cls: 0 for cls in DST_DIRS}
    skipped = 0

    for img_path in sorted(src_dir.glob('*.jpg')):
        matched_class = None
        for prefix, cls in GROUP_CLASS_MAP.items():
            if img_path.stem.startswith(prefix):
                matched_class = cls
                break

        if matched_class is None:
            skipped += 1
            continue

        dst_path = DST_DIRS[matched_class] / img_path.name
        shutil.copy(img_path, dst_path)
        copied[matched_class] += 1

    print("커스텀 데이터 복사 완료!")
    for cls, count in copied.items():
        print(f"  {cls:8s}: {count}장")
    if skipped:
        print(f"  (미분류 스킵: {skipped}장)")


if __name__ == '__main__':
    process_custom_data()
