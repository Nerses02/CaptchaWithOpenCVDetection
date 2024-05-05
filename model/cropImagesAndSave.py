import cv2
import os

def save_image_crop(img, rect, save_path):
    """ Saves the cropped area of the image defined by rect, resizing it to 250x250. """
    x, y, w, h = rect
    cropped_image = img[y:y+h, x:x+w]
    resized_image = cv2.resize(cropped_image, (250, 250), interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, resized_image)

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img, square_dim
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        #ix, iy = x, y  # Capture the starting point
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Update the rectangle while keeping it within the image boundaries
            ix = max(0, min(x, img.shape[1] - square_dim))
            iy = max(0, min(y, img.shape[0] - square_dim))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # Reset the drawing state on button release

def resize_image_aspect_ratio(img, max_width, max_height):
    """ Resize image maintaining aspect ratio. """
    height, width = img.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

def main():
    global ix, iy, drawing, img, square_dim
    drawing = False  # Initial state of drawing should be False
    image_folder = 'static/other'  # Replace with the path to your image folder
    save_folder = 'static/otherCropped'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    images = [f'image_{i}.jpg' for i in range(690)]  # Assuming you have 690 images named sequentially

    for image_name in images:
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)
        img_path = os.path.join(image_folder, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = resize_image_aspect_ratio(img, 800, 600)

        # Reset for each new image
        square_dim = min(img.shape[0], img.shape[1])
        ix, iy = 0, 0  # Start the rectangle at top-left corner for each new image
        drawing = False  # Ensure drawing is reset for each new image

        while True:
            clone = img.copy()
            cv2.rectangle(clone, (ix, iy), (ix + square_dim, iy + square_dim), (0, 255, 0), 2)
            cv2.imshow('Image', clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                crop_save_path = os.path.join(save_folder, f'{image_name}')
                save_image_crop(img, (ix, iy, square_dim, square_dim), crop_save_path)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
