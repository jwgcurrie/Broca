import cv2
import numpy as np
from PIL import Image

class Viewer:
    """
    Handles displaying images in a window using OpenCV.
    """
    def __init__(self, window_name="Captured Image"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show_image(self, image: Image.Image):
        """
        Displays a PIL image in the OpenCV window.
        """
        # Convert PIL Image to NumPy array
        frame_rgb = np.array(image)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, frame_bgr)
        cv2.waitKey(1)  # Allows the window to be updated

    def cleanup(self):
        """
        Destroys the display window.
        """
        cv2.destroyWindow(self.window_name)
