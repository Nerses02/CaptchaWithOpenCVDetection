import cv2
import numpy as np

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def create_color_histogram(image, bins=8):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_img], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv_img], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv_img], [2], None, [bins], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    hist_features = np.hstack([hist_h, hist_s, hist_v])
    return hist_features

# Load the KNN model
knn = cv2.ml.KNearest_load("model/colorKnnModelCatsAndPhotocamera.xml")

# Load and preprocess the image
img = cv2.imread('static/cat/image_24.jpg')
if img is None:
    print("Error loading image")
    exit(1)

img = resize_image(img, 150)  # Resizing to make processing faster and fit screen
original_img = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find and filter contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 1000]  # Filter out small contours

# Object detection and recognition
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if not(w > 20 and h > 100):
        continue
    roi = original_img[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (50, 50))  # Resize as per model training
    roi_features = create_color_histogram(roi_resized)

    ret, result, neighbors, dist = knn.findNearest(np.array([roi_features]), k=5)
    label_num = int(result[0][0])
    if label_num == 1:
        label = "Camera" 
    elif label_num == 2:
        label = "Cat"
    else:
        label = "Other"

    cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(original_img, label, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow('Object Detection', original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
