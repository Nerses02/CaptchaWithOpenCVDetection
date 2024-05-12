import cv2
import numpy as np

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

def find_group_bounding_box(image_path, proximity_threshold, min_width, min_height, knn_model_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    # Load the trained KNN model
    knn = cv2.ml.KNearest_load(knn_model_path)

    # Convert to grayscale and apply GaussianBlur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold bounding boxes and centers
    bounding_boxes = []

    # Calculate bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Find close contours and group them
    groups = []
    used = set()

    for i in range(len(bounding_boxes)):
        if i in used:
            continue
        group = [bounding_boxes[i]]
        queue = [i]
        while queue:
            current = queue.pop(0)
            for j in range(len(bounding_boxes)):
                if j in used or j == current:
                    continue
                if is_close(bounding_boxes[current], bounding_boxes[j], proximity_threshold):
                    queue.append(j)
                    group.append(bounding_boxes[j])
                    used.add(j)
        groups.append(group)
        used.add(i)

    # Draw bounding box for each group
    contour_img = img.copy()
    for group in groups:
        x_min = min([box[0] for box in group])
        y_min = min([box[1] for box in group])
        x_max = max([box[0] + box[2] for box in group])
        y_max = max([box[1] + box[3] for box in group])
        
        # Filter final bounding boxes based on minimum size
        if (x_max - x_min >= min_width) and (y_max - y_min >= min_height):
            cv2.rectangle(contour_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Extract the region of interest
            roi = img[y_min:y_max, x_min:x_max]
            roi_resized = cv2.resize(roi, (50, 50))  # Resize as per model training
            roi_features = create_color_histogram(roi_resized)

            k = 5
            ret, result, neighbors, dist = knn.findNearest(np.array([roi_features]), k)

            label_num = int(result[0][0])
            num_votes = np.sum(neighbors == label_num)
            confidence = (num_votes / k) * 100
            if confidence < 50:
                continue
            label = "other"
            if label_num == 1:
                label = f'Camera {confidence}%'
            elif label_num == 2:
                label = f'Cat {confidence}%'
            else:
                label = f'other {confidence}%'

            cv2.putText(contour_img, label, (x_min+5, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display image
    cv2.imshow('Grouped Contours with Labels', contour_img)

    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def is_close(box1, box2, threshold):
    # Calculate center of each box
    center1 = (box1[0] + box1[2] // 2, box1[1] + box1[3] // 2)
    center2 = (box2[0] + box2[2] // 2, box2[1] + box2[3] // 2)
    # Calculate distance
    dist = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return dist < threshold

# Usage example
find_group_bounding_box('static/cat/image_18.jpg', 50, 70, 70, "model/colorKnnModelCatsAndPhotocamera.xml")
