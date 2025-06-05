from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            circumference = cv2.arcLength(cnt, True)
            results.append({
                'width': w,
                'height': h,
                'circumference': round(circumference, 2)
            })
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    dimensions = []
    image_url = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            dimensions = analyze_image(filepath)
            image_url = filepath
    return render_template('index.html', dimensions=dimensions, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
