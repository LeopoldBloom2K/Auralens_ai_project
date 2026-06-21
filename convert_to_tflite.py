import tensorflow as tf
import onnx2tf
import onnx
import numpy as np
import os
import shutil


def _simplify_onnx(onnx_path: str, simplified_path: str) -> str:
    """onnxsim으로 그래프를 단순화. 실패하면 원본 경로 반환."""
    try:
        import onnxsim
        model = onnx.load(onnx_path)
        simplified, ok = onnxsim.simplify(model)
        if ok:
            onnx.save(simplified, simplified_path)
            print(f"  ONNX 단순화 완료: {simplified_path}")
            return simplified_path
        else:
            print("  ONNX 단순화 실패 — 원본 사용")
    except Exception as e:
        print(f"  ONNX 단순화 스킵 ({e}) — 원본 사용")
    return onnx_path


def convert_onnx_to_tflite():
    onnx_path      = './weights/auralens_model.onnx'
    simplified_path = './weights/auralens_model_simplified.onnx'
    tf_model_path  = './weights/tf_saved_model'
    tflite_path    = './weights/auralens_model_int8.tflite'

    if not os.path.exists(onnx_path):
        print("ONNX 파일을 찾을 수 없습니다. export_tflite.py를 먼저 실행하세요.")
        return

    # 이전 SavedModel 캐시 삭제 (오래된 변환 결과가 남아 있으면 오류 유발)
    if os.path.exists(tf_model_path):
        shutil.rmtree(tf_model_path)

    # ---------------------------------------------------------
    # 0단계: ONNX 그래프 단순화 (RESHAPE 등 복잡한 노드 정리)
    # ---------------------------------------------------------
    print("0 ONNX 그래프 단순화 중...")
    src_onnx = _simplify_onnx(onnx_path, simplified_path)

    # ---------------------------------------------------------
    # 1단계: onnx2tf — ONNX → TensorFlow SavedModel
    # ---------------------------------------------------------
    print("\n1 ONNX -> TF SavedModel 변환 중...")
    onnx2tf.convert(
        input_onnx_file_path=src_onnx,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    print(f"  TF SavedModel 저장 완료: {tf_model_path}")

    # ---------------------------------------------------------
    # 2단계: TFLite 동적 범위 양자화 변환 (가중치 INT8, 활성화 float32)
    # 캘리브레이션 데이터 없이도 full INT8 대비 동등한 크기 감소 달성.
    # full INT8(활성화까지 양자화)은 onnx2tf 변환 모델에서 출력 스케일
    # 불일치를 일으키므로 사용하지 않음.
    # ---------------------------------------------------------
    print("\n2 TFLite 동적 범위 양자화 변환 중...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"\nTFLite INT8 모델 완성: {tflite_path}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    convert_onnx_to_tflite()
