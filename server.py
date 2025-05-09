from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# 加载YOLOv11模型
model = YOLO('yolo11n.pt')  # 假设已下载预训练模型或使用自定义训练模型[2,5](@ref)

def detect_person_count(image_path):
    # 执行推理
    results = model.predict(image_path, save=False, verbose=False)
    
    # 统计人形数量
    person_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # 假设YOLOv11中0类为人形
                person_count += 1
                
    return person_count

@app.route('/detect', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # 保存上传的图片
    upload_path = os.path.join('uploads', file.filename)
    file.save(upload_path)
    
    try:
        # 执行检测
        count = detect_person_count(upload_path)
        result = {"person_count": count, "over_one": count > 1}
        
        # 清理临时文件
        os.remove(upload_path)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 创建上传目录
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)