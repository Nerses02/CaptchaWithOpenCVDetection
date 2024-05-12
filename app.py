from flask import Flask, render_template, request, jsonify
import os
import random
import cv2
import numpy as np

app = Flask(__name__)

# Assume images are stored in static/cat, static/photocamera, static/other
categories = ['cat', 'photocamera', 'other']
instructions = ['Please click each image containing a camera. If there are none, click Skip.', 
                'Please click each image containing a cat. If there are none, click Skip.']
instructionIndex = 0
used_images = set()
selected_images = []  # List to store selected images
knn = cv2.ml.KNearest_load('model/colorKnnModelCatsAndPhotocamera.xml')

def get_random_images():
    images = []
    used_images.clear()
    while len(images) < 9:
        category = random.choice(categories)
        directory_path = f'static/{category}'
        # List only files, excluding directories and ensuring they are not in used_images
        image_files = [f for f in os.listdir(directory_path) 
                       if os.path.isfile(os.path.join(directory_path, f)) 
                       and f"{category}/{f}" not in used_images]
        if image_files:
            random_image = random.choice(image_files)
            image_path = f"{category}/{random_image}"
            images.append(image_path)
            used_images.add(image_path)
    return images

def get_random_image():
    category = random.choice(categories)
    directory_path = f'static/{category}'
    image_files = [f for f in os.listdir(directory_path) 
                    if os.path.isfile(os.path.join(directory_path, f)) 
                    and f"{category}/{f}" not in used_images]
    if image_files:
        random_image = random.choice(image_files)
        image_path = f"{category}/{random_image}"
        used_images.add(image_path)
    return image_path

def create_color_histogram(image_path, bins=8):
    img = cv2.imread(image_path)
    if img is None:
        return None
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    hist_features = np.hstack([hist_h, hist_s, hist_v])
    return hist_features

@app.route('/')
def home():
    images = get_random_images()
    return render_template('index.html', images=images)

@app.route('/reload_images', methods=['POST'])
def reload_images():
    images = get_random_images()
    return jsonify(images=images)

@app.route('/load_single_image', methods=['POST'])
def load_single_image():
    image = get_random_image()
    return jsonify(image=image)

@app.route('/image_click', methods=['POST'])
def image_click():
    image_data = request.json['image']
    selected_images.append(image_data)  # Save the path of the clicked image
    return jsonify(success=True)

@app.route('/load_instruction', methods=['POST'])
def load_instruction():
    global instructionIndex
    instruction = random.choice(instructions)
    instructionIndex = instructions.index(instruction)
    return jsonify(message=instruction)

@app.route('/submit_images', methods=['POST'])
def submit_images():
    activ_images = request.json.get('activImages', [])

    ok = True
    for imagePath in selected_images:
        imageColorHistogram = create_color_histogram(imagePath)
        if imageColorHistogram is not None:
            imageColorHistogram = imageColorHistogram.reshape(1, -1)

            ret, result, neighbours, dist = knn.findNearest(imageColorHistogram, k=3)

            ok = ok and (result[0][0] == instructionIndex+1)
        else:
            print("Failed to process the test image.")
    if ok:
        messageForPrint = "Successful"
        for activImagePath in activ_images:
            imageColorHistogram = create_color_histogram(activImagePath)
            if imageColorHistogram is not None:
                imageColorHistogram = imageColorHistogram.reshape(1, -1)

                ret, result, neighbours, dist = knn.findNearest(imageColorHistogram, k=3)

                ok = ok and (result[0][0] != instructionIndex+1)
                if not ok:
                    messageForPrint = "Make sure you have selected all the matching pictures"
            else:
                print("Failed to process the test image.")
    else:
        messageForPrint = "Failed"
    
    selected_images.clear()

    return jsonify(message=messageForPrint)

if __name__ == "__main__":
    app.run(debug=True)