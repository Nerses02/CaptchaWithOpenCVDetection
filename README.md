
# Captcha with OpenCV Detection

This project implements a captcha system integrated with OpenCV to detect objects within images. It is designed to test captcha functionality with a mechanism for detecting specific objects, enhancing security measures by requiring users to identify images containing certain items.

## Features

- Dynamic image captcha generation.
- Object detection using OpenCV.
- Interactive web interface.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.6+
- Flask
- OpenCV for Python
- NumPy

You can install the necessary Python packages with:
```
pip install flask opencv-python numpy
```

## Project Structure

```
project/
│
├── model/              # object detection model and tools 
├── app.py              # Backend Flask application
├── index.html          # Frontend HTML file
├── static/             # Static files directory
│   └── background.png  # Background image for the webpage
│
└── README.md           # Documentation file
```

## Setup and Running the Application

1. **Clone the Repository:**
   ```
   git clone https://github.com/Nerses02/CaptchaWithOpenCVDetection.git
   cd YourRepositoryName
   ```

2. **Start the Flask App:**
   ```
   python app.py
   ```
   This command will start the Flask server on `localhost` and typically on port `5000`.

3. **Access the Web Interface:**
   Open a browser and go to `http://localhost:5000/` to interact with the captcha system.

## How It Works

- The webpage loads with a grid of images.
- Users interact by clicking images that match the instruction (e.g., "click all images with cats").
- The system validates selections based on predefined object detection criteria.

## Contributing

Contributions to the project are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Contact

Nerses Manukyan - YourEmail@example.com

Project Link: [https://github.com/Nerses02/CaptchaWithOpenCVDetection.git](https://github.com/Nerses02/CaptchaWithOpenCVDetection.git)
