import cv2
import numpy as np
import os

def create_color_histogram(image, bins=8):
    # Load the image
    img = cv2.imread(image)
    if img is None:
        return None
    
    # Convert to a desired color space (HSV in this case)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])  # Hue channel
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])  # Saturation channel
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])  # Value channel
    
    # Normalize and flatten the histograms
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    # Concatenate histograms into a single feature
    hist_features = np.hstack([hist_h, hist_s, hist_v])
    return hist_features

# Example training data preparation
features = []
labels = []  # You need to define labels for your images

entries = os.listdir("static/photocamera")
imageFiles = [entry for entry in entries if os.path.isfile(os.path.join("static/photocamera", entry))]
for img in imageFiles:
    histogram = create_color_histogram(f"static/photocamera/{img}")
    if histogram is not None:
        features.append(histogram)
        labels.append("1")

entries = os.listdir("static/cat")
imageFiles = [entry for entry in entries if os.path.isfile(os.path.join("static/cat", entry))]
for img in imageFiles:
    histogram = create_color_histogram(f"static/cat/{img}")
    if histogram is not None:
        features.append(histogram)
        labels.append("2")

entries = os.listdir("static/other")
imageFiles = [entry for entry in entries if os.path.isfile(os.path.join("static/other", entry))]
for img in imageFiles:
    histogram = create_color_histogram(f"static/other/{img}")
    if histogram is not None:
        features.append(histogram)
        labels.append("3")

# Convert lists to numpy arrays
features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Train KNN
knn = cv2.ml.KNearest_create()
knn.train(features, cv2.ml.ROW_SAMPLE, labels)

# Save the model
knn.save('model/colorKnnModelCatsAndPhotocamera.xml')
