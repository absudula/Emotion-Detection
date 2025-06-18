# display_module.py
"""
Manages the display of camera feed and cGAN output in separate OpenCV windows.
"""
import cv2
import numpy as np

class DisplayManager:
    """
    Handles creating and updating OpenCV windows for the application.
    """
    def __init__(self, window_camera_name="Camera Feed", window_cgan_name="cGAN Output"):
        """
        Initializes the display manager and creates named windows.

        Args:
            window_camera_name (str): Name for the camera feed window.
            window_cgan_name (str): Name for the cGAN output window.
        """
        self.window_camera_name = window_camera_name
        self.window_cgan_name = window_cgan_name

        cv2.namedWindow(self.window_camera_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.window_cgan_name, cv2.WINDOW_AUTOSIZE) # Or cv2.WINDOW_NORMAL for resizable
        print(f"Display windows created: '{self.window_camera_name}' and '{self.window_cgan_name}'")

    def show_camera_feed(self, frame, expressions_data):
        """
        Displays the camera feed with bounding boxes and emotion labels.

        Args:
            frame (numpy.ndarray): The camera frame to display.
            expressions_data (list): A list of dictionaries, where each dict contains
                                     'box' (x,y,w,h) and 'emotion' (str).
        """
        if frame is None:
            # Create a black screen if frame is None
            # Get window size if possible, or use a default
            h, w = 480, 640 # Default size
            # Note: cv2.getWindowImageRect is not universally available or might return -1
            # For robustness, use a default or pre-defined size for the placeholder
            display_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(display_frame, "No camera input", (50, h // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            display_frame = frame.copy()
            for expr_info in expressions_data:
                x, y, w, h = expr_info["box"]
                emotion_text = expr_info["emotion"]
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Prepare text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (0, 0, 0) # Black text
                bg_color = (0, 255, 0) # Green background for text

                # Get text size to draw a background rectangle
                (text_w, text_h), baseline = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)
                
                # Put text background
                text_y_pos = y - 10
                cv2.rectangle(display_frame, (x, text_y_pos - text_h - baseline//2), (x + text_w, text_y_pos + baseline//2), bg_color, -1)
                # Put text
                cv2.putText(display_frame, emotion_text, (x, text_y_pos), font, font_scale, text_color, font_thickness)

        cv2.imshow(self.window_camera_name, display_frame)

    def show_cgan_output(self, cgan_image, target_size=(256, 256)):
        """
        Displays the cGAN generated image.

        Args:
            cgan_image (numpy.ndarray or None): The cGAN output image.
            target_size (tuple): The (width, height) to display the cGAN image.
        """
        if cgan_image is None or cgan_image.size == 0:
            # Display a placeholder if no cGAN image is available
            placeholder = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            cv2.putText(placeholder, "cGAN: Waiting...", (target_size[0]//2 - 100, target_size[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow(self.window_cgan_name, placeholder)
        else:
            # Resize if the cGAN image is not already the target display size
            if cgan_image.shape[0:2] != (target_size[1], target_size[0]):
                display_image = cv2.resize(cgan_image, target_size, interpolation=cv2.INTER_NEAREST)
            else:
                display_image = cgan_image
            cv2.imshow(self.window_cgan_name, display_image)

    def destroy_windows(self):
        """
        Closes all OpenCV windows.
        """
        cv2.destroyAllWindows()
        print("Display windows destroyed.")

if __name__ == '__main__':
    # Example Usage
    display_mgr = DisplayManager()
    
    # Simulate camera feed
    mock_camera_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mock_expressions = [
        {"box": (100, 100, 150, 150), "emotion": "happy"},
        {"box": (300, 150, 100, 120), "emotion": "surprise"}
    ]
    
    # Simulate cGAN output
    mock_cgan_output = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    cv2.putText(mock_cgan_output, "Mock Joy", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)

    print("Displaying mock data for 5 seconds. Press 'q' to quit earlier.")
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 5.0: # Run for 5 seconds
        display_mgr.show_camera_feed(mock_camera_frame, mock_expressions)
        display_mgr.show_cgan_output(mock_cgan_output)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
            
    display_mgr.destroy_windows()
    print("DisplayManager test finished.")

