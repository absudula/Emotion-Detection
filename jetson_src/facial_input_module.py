# facial_input_module.py
"""
Handles camera input for the application.
"""
import cv2

class CameraInput:
    """
    A class to manage camera input using OpenCV.
    It can handle both standard camera IDs and GStreamer pipelines.
    """
    def __init__(self, camera_source):
        """
        Initializes the camera.

        Args:
            camera_source (int or str): The camera ID (e.g., 0) or a GStreamer pipeline string.
        """
        self.camera_source = camera_source
        if isinstance(self.camera_source, str) and "nvarguscamerasrc" in self.camera_source:
            # Using GStreamer pipeline, likely for a CSI camera on Jetson
            self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_GSTREAMER)
            print(f"Attempting to open GStreamer pipeline: {self.camera_source}")
        else:
            # Using a standard camera ID (e.g., for a USB camera)
            self.cap = cv2.VideoCapture(self.camera_source)
            print(f"Attempting to open camera ID: {self.camera_source}")

        if not self.cap.isOpened():
            error_message = f"Error: Cannot open camera source: {self.camera_source}. "
            if isinstance(self.camera_source, str):
                error_message += "Check GStreamer pipeline and camera connection."
            else:
                error_message += "Check camera connection and permissions."
            raise IOError(error_message)
        
        print(f"Successfully opened camera: {self.camera_source}")


    def get_frame(self):
        """
        Reads a frame from the camera.

        Returns:
            numpy.ndarray or None: The captured frame, or None if a frame cannot be read.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            return None
        return frame

    def release(self):
        """
        Releases the camera capture object.
        """
        if self.cap.isOpened():
            self.cap.release()
            print("Camera released.")

if __name__ == '__main__':
    # Example usage (test this on your Jetson)
    # Replace with your actual camera ID or GStreamer pipeline
    # For USB camera:
    # CAMERA_SOURCE_TO_TEST = 0
    # For Jetson CSI camera (example, adjust parameters as needed):
    CAMERA_SOURCE_TO_TEST = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! " # Smaller resolution for display
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    
    try:
        camera = CameraInput(CAMERA_SOURCE_TO_TEST)
        print("Camera initialized for testing. Press 'q' to quit.")
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("No frame received, exiting test.")
                break
            
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
    except IOError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


