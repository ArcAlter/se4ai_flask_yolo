from flask import Flask, render_template, request
from PIL import Image
import io
from ultralytics import YOLO
import base64
import cv2

model = YOLO('yolo11n.pt')

app = Flask(__name__)

@app.route("/test_flask")
def main():
    return render_template('hello.html')

# @app.route("/predict", methods=['POST'])
# def predict():
#     img = request.files['image_upload']
#     if img:
#         image = Image.open(io.BytesIO(img.read())).convert('RGB')
#         rs = model.predict(image)

#         #ส่งรูป bounding box กลับไป user
#         im = rs[0].plot()
#         res_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         rs_img = Image.fromarray(res_rgb)
#         buffered = io.BytesIO()
#         rs_img.save(buffered, format="JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

#         #Send text of model
#         detect = []
#         if rs[0].boxes is None or len(rs[0].boxes) == 0:
#             detect.append({
#                 "class": "Not found",
#                 "conf": None
#             })
#         else:
#             for cls_id, conf in zip(rs[0].boxes.cls, rs[0].boxes.conf):
#                 detect.append({
#                     "class": rs[0].name[int(cls_id)],
#                     "conf": float(conf)
#                 })
#     return render_template("result.html", detections=detect, image=img_str)

@app.route("/predict", methods=['POST'])
def predict():
    img = request.files['image_upload']
    if img:
        # อ่านรูปภาพ
        image = Image.open(io.BytesIO(img.read())).convert('RGB')
        
        # ทำการ Prediction
        rs = model.predict(image, conf=0.8)

        # 1. วาด Bounding Box (อย่าลืมเติมวงเล็บหลัง plot)
        im = rs[0].plot() 
        
        # 2. แปลง BGR (OpenCV) เป็น RGB (PIL) เพื่อส่งไปแสดงผลบนเว็บ
        res_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rs_img = Image.fromarray(res_rgb)
        
        # 3. แปลงรูปภาพเป็น Base64 string
        buffered = io.BytesIO()
        rs_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # จัดเตรียมข้อมูล Text เพื่อส่งกลับไปแสดงผล
        detect = []
        if rs[0].boxes is None or len(rs[0].boxes) == 0:
            detect.append({
                "class": "Not found",
                "conf": None
            })
        else:
            for cls_id, conf in zip(rs[0].boxes.cls, rs[0].boxes.conf):
                detect.append({
                    # แก้ไขตรงนี้: จาก .name เป็น .names
                    "class": rs[0].names[int(cls_id)], 
                    "conf": float(conf)
                })
                
    return render_template("result.html", detections=detect, image=img_str)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2569, debug=True)