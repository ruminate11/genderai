from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from detect import detect_gender_age  # Importing the detect_gender_age function from detect.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Make sure index.html is present in the 'templates' folder

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    # Decode the base64 image
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame with detect_gender_age function from detect.py
    processed_frame, results = detect_gender_age(frame)
    
    # Encode the processed frame back to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_data = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': 'data:image/jpeg;base64,' + processed_data})

if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port)
  # Change debug=False when deploying to production
