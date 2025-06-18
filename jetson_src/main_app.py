#!/usr/bin/env python3
# main_app.py
"""
Main Application leveraging TensorRT 10.3.0 for inference acceleration on NVIDIA Jetson Orin Nano.
"""

import os
import sys
import time
import threading
import queue
import signal
import numpy as np
from collections import deque
import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced custom modules
from facial_input_module import CameraInput
from facial_expression_tensorrt_module import TensorRTFacialExpressionRecognizer, initialize_cuda_context
from cgan_tensorrt_module import TensorRTCGANAnimator
from display_module import DisplayManager
from decision_system_module import PlayfulResponder

# Configuration
CAMERA_SOURCE = 0  # USB camera

# Model paths - UPDATE THESE TO YOUR ACTUAL PATHS
FER_TENSORRT_ENGINE_PATH = "models/patt_lite_enhanced.trt"  # TensorRT engine for PAtt-Lite
FER_ONNX_MODEL_PATH = "models/patt_lite_enhanced.onnx"  # ONNX fallback for PAtt-Lite
CGAN_TENSORRT_ENGINE_PATH = "models/cgan_generator.trt"  # TensorRT engine for cGAN
CGAN_TF_WEIGHTS_PATH = "models/best_generator.weights.h5"  # TensorFlow fallback for cGAN

# Window names
WINDOW_CAMERA_NAME = "Facial Expression Feed"
WINDOW_CGAN_NAME = "AI Response"

# PAtt-Lite emotion labels (from training script)
PATT_LITE_EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
CGAN_EMOTIONS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

# Emotion mapping from PAtt-Lite to cGAN
EMOTION_MAPPING = {
    'Anger': 'anger',
    'Disgust': 'disgust', 
    'Fear': 'fear',
    'Happiness': 'happy',
    'Sadness': 'sadness',
    'Surprise': 'surprise',
    'Neutral': 'neutral'
}

class FrameRateLimiter:
    """Proper frame rate limiter using high-resolution timing"""
    
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        self.last_frame_time = time.time()
        
    def wait(self):
        """Wait to maintain target frame rate"""
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        sleep_time = self.frame_duration - elapsed
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        self.last_frame_time = time.time()

class PerformanceMonitor:
    """Accurate performance monitoring for real-time systems"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.total_frames = 0
        self.start_time = time.time()
        
    def update_processing_time(self, processing_time):
        """Update with actual end-to-end processing time"""
        self.processing_times.append(processing_time)
        self.total_frames += 1
    
    def update_inference_time(self, inference_time):
        """Update with inference-only time"""
        self.inference_times.append(inference_time)
    
    def get_stats(self):
        """Get accurate performance statistics"""
        stats = {'total_frames': self.total_frames}
        
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            stats['fps'] = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
            stats['avg_processing_time_ms'] = avg_processing_time * 1000
            
        if self.inference_times:
            stats['avg_inference_time_ms'] = np.mean(self.inference_times) * 1000
            
        # Overall runtime statistics
        runtime = time.time() - self.start_time
        stats['runtime_fps'] = self.total_frames / runtime if runtime > 0 else 0
        
        return stats

class EnhancedOptimizedEmotionApp:
    """
    Enhanced optimized emotion detection application for TensorRT 10.3.0
    Updated for PAtt-Lite model with therapeutic animated responses and smooth animation
    """
    
    def __init__(self, target_fps=30, therapeutic_mode=True, enable_smooth_animation=True):
        self.camera = None
        self.expression_recognizer = None
        self.cgan_animator = None
        self.display_manager = None
        self.playful_responder = None
        
        # Configuration
        self.target_fps = target_fps
        self.therapeutic_mode = therapeutic_mode
        self.enable_smooth_animation = enable_smooth_animation
        
        # Frame rate control
        self.frame_limiter = FrameRateLimiter(target_fps)
        
        print(f"Target FPS: {target_fps} ({1000/target_fps:.1f}ms per frame)")
        print(f"Therapeutic mode: {'ENABLED' if therapeutic_mode else 'DISABLED (exact match)'}")
        print(f"Smooth animation: {'ENABLED' if enable_smooth_animation else 'DISABLED'}")
        
        # Threading components
        self.camera_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        
        # Performance monitoring (without GPU utilization)
        self.perf_monitor = PerformanceMonitor()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("Enhanced Emotion App initialized for PAtt-Lite with therapeutic responses")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
    
    def toggle_therapeutic_mode(self):
        """Toggle between therapeutic and exact match modes"""
        self.therapeutic_mode = not self.therapeutic_mode
        if self.playful_responder:
            self.playful_responder.set_exact_match_mode(not self.therapeutic_mode)
        
        mode_text = "THERAPEUTIC" if self.therapeutic_mode else "EXACT MATCH"
        print(f"Switched to {mode_text} mode")
    
    def toggle_smooth_animation(self):
        """Toggle smooth animation on/off"""
        self.enable_smooth_animation = not self.enable_smooth_animation
        if self.cgan_animator:
            self.cgan_animator.set_smooth_animation(self.enable_smooth_animation)
        
        print(f"Smooth animation: {'ENABLED' if self.enable_smooth_animation else 'DISABLED'}")
    
    def initialize_components(self):
        """Initialize all components with enhanced features"""
        try:
            print("Initializing enhanced components for TensorRT 10.3.0 with PAtt-Lite...")
            
            # CRITICAL: Initialize global CUDA context FIRST
            print("Initializing global CUDA context...")
            initialize_cuda_context()
            
            # Camera input
            self.camera = CameraInput(CAMERA_SOURCE)
            
            # Enhanced TensorRT FER with PAtt-Lite
            print("Initializing Enhanced TensorRT FER with PAtt-Lite...")
            self.expression_recognizer = TensorRTFacialExpressionRecognizer(
                tensorrt_engine_path=FER_TENSORRT_ENGINE_PATH,
                onnx_model_path=FER_ONNX_MODEL_PATH,
                emotion_labels=PATT_LITE_EMOTIONS,
                enable_caching=True,
                max_cache_size=5
            )
            self.expression_recognizer.enable_performance_profiling(True)
            
            # Enhanced TensorRT cGAN with smooth animation
            print("Initializing Enhanced TensorRT cGAN with smooth animation...")
            self.cgan_animator = TensorRTCGANAnimator(
                tensorrt_engine_path=CGAN_TENSORRT_ENGINE_PATH,
                tensorflow_weights_path=CGAN_TF_WEIGHTS_PATH,
                enable_smooth_animation=self.enable_smooth_animation
            )
            
            # Display manager
            self.display_manager = DisplayManager(WINDOW_CAMERA_NAME, WINDOW_CGAN_NAME)
            
            # Enhanced decision system with therapeutic sequences
            self.playful_responder = PlayfulResponder(
                CGAN_EMOTIONS, 
                exact_match_mode=not self.therapeutic_mode
            )
            
            # Print enhanced component info
            fer_stats = self.expression_recognizer.get_performance_stats()
            cgan_info = self.cgan_animator.get_performance_info()
            
            print(f"All enhanced components initialized successfully")
            print(f"FER Backend: {fer_stats.get('backend', 'Unknown')}")
            print(f"FER Input Size: {fer_stats.get('input_size', 'Unknown')}")
            print(f"cGAN Backend: {cgan_info['backend']}")
            print(f"Smooth Animation: {cgan_info['smooth_animation']}")
            print(f"Interpolation Steps: {cgan_info['interpolation_steps']}")
            
            return True
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def camera_thread(self):
        """Camera capture thread - provides frames to processing"""
        print(f"Camera thread started")
        
        while self.running:
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    try:
                        self.camera_queue.put(frame, block=False)
                    except queue.Full:
                        # Drop oldest frame and add new one
                        try:
                            self.camera_queue.get_nowait()
                            self.camera_queue.put(frame, block=False)
                        except queue.Empty:
                            pass
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                    
            except Exception as e:
                print(f"Camera thread error: {e}")
                time.sleep(0.1)
        
        print("Camera thread stopped")
    
    def processing_thread(self):
        """Main processing thread with enhanced therapeutic response"""
        print(f"Processing thread started (target: {self.target_fps} FPS)")
        
        while self.running:
            processing_start_time = time.time()
            
            try:
                # Get frame from camera queue
                try:
                    frame = self.camera_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Inference timing
                inference_start = time.time()
                
                # Facial expression recognition with PAtt-Lite
                expressions = self.expression_recognizer.detect_and_recognize_optimized(frame)
                
                # Process emotions for enhanced cGAN response
                cgan_emotion_name = 'neutral'
                if expressions:
                    detected_emotion = expressions[0]['emotion']
                    cgan_emotion_name = EMOTION_MAPPING.get(detected_emotion, 'neutral')
                
                # Get enhanced response from decision system (therapeutic or exact match)
                cgan_emotion_idx, sequence_desc, changed = self.playful_responder.get_animator_response(cgan_emotion_name)
                
                # Generate enhanced cGAN image with smooth animation
                cgan_image = self.cgan_animator.generate_image(cgan_emotion_idx)
                
                # Calculate inference time
                inference_time = time.time() - inference_start
                
                # Package results with enhanced info
                result_data = {
                    'frame': frame,
                    'expressions': expressions,
                    'cgan_image': cgan_image,
                    'sequence_desc': sequence_desc,
                    'inference_time': inference_time,
                    'timestamp': time.time(),
                    'therapeutic_mode': self.therapeutic_mode,
                    'animation_status': self.cgan_animator.get_animation_status(),
                    'responder_status': self.playful_responder.get_status_info()
                }
                
                # Send to display queue
                try:
                    self.result_queue.put(result_data, block=False)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result_data, block=False)
                    except queue.Empty:
                        pass
                
                # Calculate total processing time
                processing_time = time.time() - processing_start_time
                
                # Update performance monitor
                self.perf_monitor.update_processing_time(processing_time)
                self.perf_monitor.update_inference_time(inference_time)
                
                # Frame rate limiting
                self.frame_limiter.wait()
                
            except Exception as e:
                print(f"Processing thread error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print("Processing thread stopped")
    
    def display_thread(self):
        """Enhanced display thread with mode controls"""
        print(f"Display thread started")
        print("Controls: 'q'=quit, 't'=toggle therapeutic mode, 's'=toggle smooth animation")
        
        frame_count = 0
        last_status_time = time.time()
        status_interval = 3.0
        
        while self.running:
            try:
                # Get result from queue
                try:
                    data = self.result_queue.get(timeout=0.2)
                except queue.Empty:
                    cv2.waitKey(1)
                    continue
                
                # Display results
                self.display_manager.show_camera_feed(data['frame'], data['expressions'])
                self.display_manager.show_cgan_output(data['cgan_image'])
                
                frame_count += 1
                
                # Print status periodically
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    self.print_enhanced_performance_status(frame_count, data)
                    last_status_time = current_time
                
                # Check for control keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exit key pressed")
                    self.running = False
                    break
                elif key == ord('t'):
                    self.toggle_therapeutic_mode()
                elif key == ord('s'):
                    self.toggle_smooth_animation()
                
            except Exception as e:
                print(f"Display thread error: {e}")
                time.sleep(0.1)
        
        print("Display thread stopped")
    
    def print_enhanced_performance_status(self, frame_count, latest_data):
        """Print enhanced performance status without meaningless GPU utilization"""
        stats = self.perf_monitor.get_stats()
        fer_stats = self.expression_recognizer.get_performance_stats()
        cgan_info = self.cgan_animator.get_performance_info()
        
        print("\n" + "="*60)
        print(f"ENHANCED PAtt-LITE SYSTEM STATUS (Frame: {frame_count})")
        print("="*60)
        
        # Accurate FPS measurements
        actual_fps = stats.get('fps', 0)
        runtime_fps = stats.get('runtime_fps', 0)
        fps_efficiency = (actual_fps / self.target_fps * 100) if self.target_fps > 0 else 0
        
        print(f"Performance:")
        print(f"  Target FPS: {self.target_fps}")
        print(f"  Actual FPS: {actual_fps:.1f} ({fps_efficiency:.1f}% of target)")
        print(f"  Runtime FPS: {runtime_fps:.1f}")
        
        # Timing breakdown
        if 'avg_processing_time_ms' in stats:
            print(f"  Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
        if 'avg_inference_time_ms' in stats:
            print(f"  Inference Time: {stats['avg_inference_time_ms']:.1f}ms")
        
        # System configuration status
        print(f"\nSystem Configuration:")
        print(f"  PAtt-Lite FER: {fer_stats.get('backend', 'Unknown')}")
        print(f"  cGAN Backend: {cgan_info['backend']}")
        print(f"  Response Mode: {'THERAPEUTIC' if self.therapeutic_mode else 'EXACT MATCH'}")
        print(f"  Smooth Animation: {'ENABLED' if self.enable_smooth_animation else 'DISABLED'}")
        
        # Animation status
        if 'animation_status' in latest_data:
            anim_status = latest_data['animation_status']
            print(f"\nAnimation Status:")
            print(f"  Interpolation: {anim_status['interpolation_alpha']}")
            print(f"  In Transition: {anim_status['in_transition']}")
            if anim_status['in_transition']:
                print(f"  Transition Progress: {anim_status['transition_progress']}")
        
        # Therapeutic system status
        if 'responder_status' in latest_data:
            resp_status = latest_data['responder_status']
            print(f"\nTherapeutic System:")
            print(f"  Mode: {resp_status['mode']}")
            print(f"  Child Emotion: {resp_status['current_child_emotion']}")
            print(f"  Sequence Step: {resp_status['sequence_step']}")
        
        # Queue status
        print(f"\nQueue Status:")
        print(f"  Camera Queue: {self.camera_queue.qsize()}")
        print(f"  Result Queue: {self.result_queue.qsize()}")
        
        # Performance insights
        actual_processing_time_ms = stats.get('avg_processing_time_ms', 0)
        max_possible_fps = 1000.0 / actual_processing_time_ms if actual_processing_time_ms > 0 else 0
        
        print(f"\nPerformance Analysis:")
        print(f"  Hardware Limit: {max_possible_fps:.1f} FPS")
        
        if self.target_fps <= max_possible_fps:
            efficiency_status = "Target achievable"
        else:
            efficiency_status = f"Target too high (max: {max_possible_fps:.1f})"
        print(f"  Target Efficiency: {efficiency_status}")
        
        print("="*60)
    
    def run(self):
        """Main application loop with enhanced features"""
        print("Starting enhanced application with PAtt-Lite and therapeutic responses...")
        
        if not self.initialize_components():
            print("Failed to initialize components")
            return
        
        self.running = True
        
        # Start threads
        threads = []
        
        # Camera thread
        camera_thread = threading.Thread(target=self.camera_thread, name="CameraThread")
        camera_thread.daemon = True
        camera_thread.start()
        threads.append(camera_thread)
        
        # Processing thread (this controls the actual FPS)
        processing_thread = threading.Thread(target=self.processing_thread, name="ProcessingThread")
        processing_thread.daemon = True
        processing_thread.start()
        threads.append(processing_thread)
        
        # Display thread (main thread)
        try:
            self.display_thread()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
            self.running = False
        
        # Wait for threads to finish
        print("Waiting for threads to complete...")
        for thread in threads:
            thread.join(timeout=3.0)
        
        # Cleanup
        self.cleanup()
        print("Application terminated successfully")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.camera:
                self.camera.release()
            
            if self.display_manager:
                self.display_manager.destroy_windows()
            
            if self.expression_recognizer:
                self.expression_recognizer.cleanup()
            
            if self.cgan_animator:
                self.cgan_animator.cleanup()
            
            print("Resources cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

def check_requirements():
    """Check if required files exist"""
    required_files = [
        (FER_ONNX_MODEL_PATH, "PAtt-Lite ONNX model"),
        (CGAN_TF_WEIGHTS_PATH, "cGAN TensorFlow weights")
    ]
    
    # Optional TensorRT engines
    optional_files = [
        (FER_TENSORRT_ENGINE_PATH, "PAtt-Lite TensorRT engine"),
        (CGAN_TENSORRT_ENGINE_PATH, "cGAN TensorRT engine")
    ]
    
    print("Model Files Status:")
    print("-" * 50)
    
    missing_required = []
    
    # Check required files
    for file_path, description in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"Found {description}: {size_mb:.1f} MB")
        else:
            print(f"Missing {description}: {file_path}")
            missing_required.append((file_path, description))
    
    # Check optional TensorRT engines
    tensorrt_available = 0
    for file_path, description in optional_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"Found {description}: {size_mb:.1f} MB")
            tensorrt_available += 1
        else:
            print(f"Missing {description}: {file_path} (will use fallback)")
    
    if missing_required:
        print("\nMissing required files:")
        for file_path, description in missing_required:
            print(f"   {description}: {file_path}")
        return False
    
    if tensorrt_available == 0:
        print("\nNo TensorRT engines found. System will use fallback models.")
    elif tensorrt_available == 1:
        print(f"\nPartial TensorRT acceleration available ({tensorrt_available}/2 engines)")
    else:
        print(f"\nFull TensorRT acceleration available ({tensorrt_available}/2 engines)")
    
    return True

def main():
    """Main function with enhanced therapeutic response system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Facial Expression Detection with Therapeutic Responses")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (realistic: 20-60, default: 30)")
    parser.add_argument("--therapeutic", action="store_true", default=True, help="Enable therapeutic response mode (default)")
    parser.add_argument("--exact-match", action="store_true", help="Use exact emotion matching instead of therapeutic")
    parser.add_argument("--smooth", action="store_true", default=True, help="Enable smooth animation (default)")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smooth animation")
    args = parser.parse_args()
    
    # Process arguments
    target_fps = max(15, min(60, args.fps))  # Realistic range
    therapeutic_mode = args.therapeutic and not args.exact_match
    enable_smooth = args.smooth and not args.no_smooth
    
    print("Enhanced Facial Expression Detection and Therapeutic Response System")
    print("TensorRT 10.3.0 Compatible - PAtt-Lite Enhanced - Evidence-Based Responses")
    print("=" * 70)
    print(f"Target FPS: {target_fps}")
    print(f"Response Mode: {'THERAPEUTIC (evidence-based sequences)' if therapeutic_mode else 'EXACT MATCH (mirror emotion)'}")
    print(f"Smooth Animation: {'ENABLED (video-like transitions)' if enable_smooth else 'DISABLED'}")
    print(f"PAtt-Lite Emotions: {PATT_LITE_EMOTIONS}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check requirements
    if not check_requirements():
        print("\nRequired model files are missing.")
        print("Instructions:")
        print("1. Train PAtt-Lite model using: python patt_lite_fer_train.py")
        print("2. Convert ONNX to TensorRT: python fer_onnx_tensorrt.py --onnx models/patt_lite_enhanced.onnx")
        print("3. Ensure cGAN weights are available")
        return 1
    
    print(f"\nENHANCED FEATURES:")
    print(f"- Evidence-based therapeutic sequences for child emotional development")
    print(f"- Smooth video-like animation transitions (90+ interpolation steps)")
    print(f"- Real-time mode switching: 't' for therapeutic toggle, 's' for smooth toggle")
    print(f"- Temporal smoothing to eliminate flickering")
    print(f"- Frame-rate independent animation timing")
    
    if therapeutic_mode:
        print(f"\nTHERAPEUTIC MODE ACTIVE:")
        print(f"- Validates child's emotions first (co-regulation)")
        print(f"- Guides toward emotional regulation gradually")
        print(f"- Based on child psychology research")
        print(f"- Supports emotional growth and well-being")
    else:
        print(f"\nEXACT MATCH MODE ACTIVE:")
        print(f"- Mirrors child's emotion directly")
        print(f"- Simple emotional validation")
    
    input("\nPress Enter to start enhanced system with therapeutic responses...")
    
    try:
        app = EnhancedOptimizedEmotionApp(
            target_fps=target_fps,
            therapeutic_mode=therapeutic_mode,
            enable_smooth_animation=enable_smooth
        )
        app.run()
        
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
