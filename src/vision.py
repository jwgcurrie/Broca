import time
from vlm_handler import VLMHandler
from webcam_tool import WebcamTool
from PIL import Image

class VisionModule:
    """
    A modular controller that integrates the webcam and the VLM,
    now with detailed performance logging.
    """
    def __init__(self):
        """Initialises all necessary vision components."""
        print("Initialising Vision Module...")
        self.vlm = VLMHandler()
        self.webcam = WebcamTool()
        self.vlm.to_gpu()
        self.conversation_history = []

    def get_visual_response(self, prompt: str) -> str | None:
        """
        Captures an image, gets a response from the VLM,
        and logs the duration of each step.
        """
        # --- Timing Logic Start ---
        timings = {}
        total_start_time = time.time()
        last_timestamp = total_start_time
        
        # 1. Capture Image
        image = self.webcam.capture_frame()
        timings["1. Image Capture"] = time.time() - last_timestamp
        last_timestamp = time.time()
        
        if image is None:
            print("Error: Could not get an image from the webcam.")
            return "I couldn't see anything. Please check if the camera is working."

        # 2. Resize Image
        image.thumbnail((640, 640))
        timings["2. Image Resize"] = time.time() - last_timestamp
        last_timestamp = time.time()

        # 3. VLM Inference
        response, self.conversation_history = self.vlm.get_response(
            prompt=prompt,
            image=image,
            history=self.conversation_history
        )
        timings["3. VLM Inference"] = time.time() - last_timestamp
        timings["Total Time"] = time.time() - total_start_time
        # --- Timing Logic End ---

        # Print the performance breakdown
        print("\n--- ⏱️ Performance Log ---")
        for stage, duration in timings.items():
            print(f"{stage:<20} | {duration:.4f} seconds")
        print("--------------------------\n")

        return response

    def cleanup(self):
        """Releases all hardware resources properly."""
        print("Releasing vision module resources...")
        self.vlm.to_cpu()
        self.webcam.release()

if __name__ == '__main__':
    vision = VisionModule()
    try:
        # The test call no longer needs a 'show_viewer' argument
        response = vision.get_visual_response("Describe what you see in a single sentence.")

        if response:
            print("VLM Response:")
            print(response)
    finally:
        vision.cleanup()