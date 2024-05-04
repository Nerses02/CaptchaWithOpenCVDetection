import cv2
import numpy as np
import shutil
import os

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

# Load the trained KNN model
knn = cv2.ml.KNearest_load('model/colorKnnModelCatsAndPhotocamera.xml')

entries = os.listdir("static/cat")
imageFiles = [entry for entry in entries if os.path.isfile(os.path.join("static/cat", entry))]
for img in imageFiles:
    test_image_path = f"static/cat/{img}"

    test_features = create_color_histogram(test_image_path)
    if test_features is not None:
        test_features = test_features.reshape(1, -1)

        ret, result, neighbours, dist = knn.findNearest(test_features, k=3)

        if result[0][0] != 2:
            shutil.move(test_image_path, "static/cat/notDetected")
    else:
        print(f"Failed to process the test image {test_image_path}.")


entries = os.listdir("static/photocamera")
imageFiles = [entry for entry in entries if os.path.isfile(os.path.join("static/photocamera", entry))]
for img in imageFiles:
    test_image_path = f"static/photocamera/{img}"

    test_features = create_color_histogram(test_image_path)
    if test_features is not None:
        test_features = test_features.reshape(1, -1)

        ret, result, neighbours, dist = knn.findNearest(test_features, k=3)

        if result[0][0] != 1:
            shutil.move(test_image_path, "static/photocamera/notDetected")
    else:
        print(f"Failed to process the test image {test_image_path}.")

entries = os.listdir("static/other")
imageFiles = [entry for entry in entries if os.path.isfile(os.path.join("static/other", entry))]
for img in imageFiles:
    test_image_path = f"static/other/{img}"

    test_features = create_color_histogram(test_image_path)
    if test_features is not None:
        test_features = test_features.reshape(1, -1)

        ret, result, neighbours, dist = knn.findNearest(test_features, k=3)

        if result[0][0] != 3:
            shutil.move(test_image_path, "static/other/notDetected")
    else:
        print(f"Failed to process the test image {test_image_path}.")