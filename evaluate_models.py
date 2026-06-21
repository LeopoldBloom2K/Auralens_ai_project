# -*- coding: utf-8 -*-
"""
evaluate_models.py
ONNX vs TFLite INT8 모델 정확도 비교 (전체 val set 기준)

Usage:
  python evaluate_models.py              # 전체 검증 데이터
  python evaluate_models.py --limit 300  # 클래스당 최대 300장
  python evaluate_models.py --onnx_only  # ONNX만 평가
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# 앱 기대 순서 (export_tflite.py의 _AppOrderWrapper 기준)
CLASS_NAMES      = ['person', 'food', 'scenery']
FOLDER_TO_IDX    = {'person': 0, 'food': 1, 'scenery': 2}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── 전처리 ────────────────────────────────────────────────────────────────────

def preprocess_nhwc(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────

def load_val_samples(val_dir: str, limit: int = 0):
    samples = []
    for folder_name, app_idx in FOLDER_TO_IDX.items():
        folder = Path(val_dir) / folder_name
        if not folder.exists():
            print(f"  Warning: {folder} 없음, 스킵")
            continue
        paths = sorted(
            p for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG')
            for p in folder.glob(ext)
        )
        if limit > 0:
            paths = paths[:limit]
        for p in paths:
            samples.append((str(p), app_idx))
    return samples


# ── 진행 표시 ─────────────────────────────────────────────────────────────────

def _progress(current: int, total: int):
    filled = int(30 * current / total)
    bar = '#' * filled + '-' * (30 - filled)
    print(f'\r  [{bar}] {current}/{total}', end='', flush=True)


# ── 모델 평가 ─────────────────────────────────────────────────────────────────

def evaluate_onnx(samples, onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime 미설치: pip install onnxruntime")
        return None

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    preds, labels, errors = [], [], 0

    for i, (img_path, true_label) in enumerate(samples):
        _progress(i + 1, len(samples))
        try:
            inp = preprocess_nhwc(img_path)
            # ONNX 모델은 NCHW 입력 (export_tflite.py dummy_input 기준)
            inp_nchw = np.transpose(inp, (0, 3, 1, 2))
            out = sess.run(None, {input_name: inp_nchw})
            preds.append(int(np.argmax(out[0])))
            labels.append(true_label)
        except Exception:
            errors += 1

    print()
    if errors:
        print(f"  (로드 실패 스킵: {errors}장)")
    return np.array(preds), np.array(labels)


def evaluate_tflite(samples, tflite_path: str):
    try:
        import tensorflow as tf
    except ImportError:
        print("  tensorflow 미설치: pip install tensorflow")
        return None

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    preds, labels, errors = [], [], 0

    for i, (img_path, true_label) in enumerate(samples):
        _progress(i + 1, len(samples))
        try:
            inp = preprocess_nhwc(img_path)  # TFLite는 NHWC 그대로
            interp.set_tensor(inp_d['index'], inp)
            interp.invoke()
            out = interp.get_tensor(out_d['index'])
            preds.append(int(np.argmax(out[0])))
            labels.append(true_label)
        except Exception:
            errors += 1

    print()
    if errors:
        print(f"  (로드 실패 스킵: {errors}장)")
    return np.array(preds), np.array(labels)


# ── 리포트 출력 ───────────────────────────────────────────────────────────────

def print_report(label: str, preds: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    overall_acc = 100.0 * np.sum(preds == labels) / n

    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    print(f"  전체 정확도 : {overall_acc:.2f}%  ({np.sum(preds == labels)}/{n})")

    # 클래스별 정확도
    print(f"\n  {'클래스':10s}  {'정확도':>9s}  {'맞춤':>6s}  {'전체':>6s}")
    print(f"  {'-'*40}")
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        total = int(mask.sum())
        if total == 0:
            continue
        correct = int((preds[mask] == idx).sum())
        print(f"  {name:10s}  {100*correct/total:>8.2f}%  {correct:>6d}  {total:>6d}")

    # Confusion matrix
    print(f"\n  Confusion Matrix  (행=실제 클래스, 열=예측 클래스)")
    header = f"  {'':10s}"
    for name in CLASS_NAMES:
        header += f"  {name[:7]:>7s}"
    print(header)
    for true_idx, true_name in enumerate(CLASS_NAMES):
        mask = labels == true_idx
        if mask.sum() == 0:
            continue
        row = f"  {true_name:10s}"
        for pred_idx in range(len(CLASS_NAMES)):
            count = int(((labels == true_idx) & (preds == pred_idx)).sum())
            row += f"  {count:>7d}"
        print(row)

    return overall_acc


def print_comparison(accs: dict):
    diff = accs['ONNX'] - accs['TFLite INT8']

    print(f"\n{'='*48}")
    print(f"  양자화 손실 요약")
    print(f"{'='*48}")
    print(f"  ONNX        : {accs['ONNX']:.2f}%")
    print(f"  TFLite INT8 : {accs['TFLite INT8']:.2f}%")
    print(f"  손실        : -{diff:.2f}%p")
    print()

    if diff > 5.0:
        print("  [재변환 필요]")
        print("  convert_to_tflite.py 의 image_paths[:100] 을 [:500] 으로 변경 후 재실행하세요.")
        print("  손실이 지속되면 2단계 학습 전략(feature extractor freeze) 적용을 고려하세요.")
    elif diff > 2.0:
        print("  [허용 범위, 개선 여지 있음]")
        print("  배포 전 실기기 테스트 권장. 필요 시 representative 샘플 수를 늘리세요.")
    else:
        print("  [양호] 배포 가능합니다.")


# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ONNX vs TFLite INT8 정확도 비교')
    parser.add_argument('--val_dir',   default='./dataset/val')
    parser.add_argument('--onnx',      default='./weights/auralens_model.onnx')
    parser.add_argument('--tflite',    default='./weights/auralens_model_int8.tflite')
    parser.add_argument('--limit',     type=int, default=0,
                        help='클래스당 최대 샘플 수 (기본값: 0=전체)')
    parser.add_argument('--onnx_only', action='store_true',
                        help='ONNX 모델만 평가 (TFLite 없을 때)')
    args = parser.parse_args()

    if not Path(args.val_dir).exists():
        print(f"ERROR: {args.val_dir} 폴더가 없습니다. prepare_dataset.py를 먼저 실행하세요.")
        sys.exit(1)

    limit_msg = f"클래스당 최대 {args.limit}장" if args.limit else "전체"
    print(f"\n검증 데이터 로딩 중... ({limit_msg})")
    samples = load_val_samples(args.val_dir, args.limit)

    if not samples:
        print("ERROR: 검증 샘플이 없습니다.")
        sys.exit(1)

    per_class = {c: sum(1 for _, l in samples if l == i) for i, c in enumerate(CLASS_NAMES)}
    print(f"총 {len(samples)}장  |  " + "  ".join(f"{c}={n}" for c, n in per_class.items()))

    accs = {}

    if Path(args.onnx).exists():
        print(f"\nONNX 평가 중...")
        result = evaluate_onnx(samples, args.onnx)
        if result is not None:
            accs['ONNX'] = print_report('ONNX 모델', *result)
    else:
        print(f"\nONNX 파일 없음: {args.onnx}")

    if not args.onnx_only:
        if Path(args.tflite).exists():
            print(f"\nTFLite INT8 평가 중...")
            result = evaluate_tflite(samples, args.tflite)
            if result is not None:
                accs['TFLite INT8'] = print_report('TFLite INT8 모델', *result)
        else:
            print(f"\nTFLite 파일 없음: {args.tflite}")

    if len(accs) == 2:
        print_comparison(accs)


if __name__ == '__main__':
    main()
