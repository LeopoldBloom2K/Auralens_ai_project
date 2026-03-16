import tensorflow as tf
import onnx2tf  # 🚀 최신 변환 라이브러리로 교체!
import numpy as np
from PIL import Image
import glob
import os

def convert_onnx_to_tflite():
    onnx_path = './weights/auralens_model.onnx'
    tf_model_path = './weights/tf_saved_model'
    tflite_path = './weights/auralens_model_int8.tflite'

    if not os.path.exists(onnx_path):
        print("❌ ONNX 파일을 찾을 수 없습니다. export_tflite.py를 먼저 실행해 주세요!")
        return

    # ---------------------------------------------------------
    # 1단계: onnx2tf를 사용하여 TensorFlow SavedModel로 변환
    # ---------------------------------------------------------
    print("1️⃣ ONNX 모델을 TensorFlow SavedModel로 변환 중 (onnx2tf 가동!)...")
    
    # onnx2tf 변환 실행
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_model_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True, # 화면에 불필요한 로그가 너무 많이 뜨는 것을 방지
    )
    print(f"✅ TF SavedModel 임시 저장 완료: {tf_model_path}")

    # ---------------------------------------------------------
    # 2단계: TensorFlow 모델을 TFLite (INT8 양자화)로 압축 변환
    # ---------------------------------------------------------
    print("\n2️⃣ TensorFlow 모델을 TFLite(INT8 양자화)로 변환 중... (시간이 조금 걸립니다)")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # INT8 양자화를 위한 최적화 켜기
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        # 검증(val) 폴더에서 무작위로 사진을 가져옴
        image_paths = glob.glob('./dataset/val/*/*.jpg') + glob.glob('./dataset/val/*/*.jpeg') + glob.glob('./dataset/val/*/*.png')
        np.random.shuffle(image_paths)
        
        for path in image_paths[:100]:  # 100장만 사용해도 충분함
            try:
                img = Image.open(path).convert('RGB')
                img = img.resize((224, 224))
                img_array = np.array(img, dtype=np.float32)
                
                # 훈련할 때 사용했던 PyTorch 정규화 수식
                img_array = img_array / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_array = (img_array - mean) / std
                
                # NCHW (1, 3, 224, 224) 차원으로 맞춤
                img_array = np.transpose(img_array, (2, 0, 1))
                img_array = np.expand_dims(img_array, axis=0)
                
                yield [img_array]
            except Exception as e:
                pass # 깨진 이미지가 있으면 무시하고 넘어감

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    # 변환 실행
    tflite_model = converter.convert()

    # 최종 파일 저장
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"\n🎉 최종 TFLite INT8 양자화 모델 완성! 파일 위치: {tflite_path}")
    print("📉 용량이 대폭 줄어들고 스마트폰 연산에 완벽하게 최적화되었습니다!")

if __name__ == '__main__':
    convert_onnx_to_tflite()