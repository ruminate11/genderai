from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from detect import detect_gender_age
import re

app = Flask(__name__)

@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result_img, results = detect_gender_age(frame)

    _, buffer = cv2.imencode('.jpg', result_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    img_data_url = f"data:image/jpeg;base64,{img_str}"

    return jsonify({
        "image": img_data_url,
        "prediction": results  # <-- Send predictions as well
    })

if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)
  # Change debug=False when deploying to production
