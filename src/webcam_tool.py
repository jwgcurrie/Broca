import cv2
import numpy as np
from PIL import Image

class WebcamTool:
    """
    A tool to interface with the system's webcam using OpenCV.
    """
    def __init__(self):
        self.cap = None

    def capture_frame(self):
        """
        Captures a single frame from the webcam.

        Returns:
            PIL.Image.Image or None: A PIL Image object if a frame is captured successfully,
                                     otherwise None.
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                self.cap = None
                return None

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            return None

        # Convert the OpenCV BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image

    def release(self):
        """
        Releases the webcam resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    webcam = WebcamTool()
    try:
        print("Attempting to capture a frame...")
        frame = webcam.capture_frame()
        if frame:
            print("Frame captured successfully!")
            frame.show()  # Display the image (requires Pillow to be installed)
        else:
            print("Failed to capture frame.")
    finally:
        webcam.release()
        print("Webcam released.")
