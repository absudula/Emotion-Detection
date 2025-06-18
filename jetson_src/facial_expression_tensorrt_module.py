# facial_expression_tensorrt_module.py
"""
TensorRT-based Facial Expression Recognition Module
"""
import cv2
import numpy as np
import time
from collections import defaultdict
from threading import Lock
import os

# Check TensorRT availability first
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # DO NOT import pycuda.autoinit - this causes threading issues
    TENSORRT_AVAILABLE = True
    print(f"TensorRT version: {trt.__version__}")
except ImportError:
    print("Warning: TensorRT/PyCUDA not available.")
    TENSORRT_AVAILABLE = False

# Check ONNX Runtime availability
try:
    import onnxruntime
    ONNX_AVAILABLE = True
    print(f"ONNX Runtime available")
except ImportError:
    print("Warning: ONNX Runtime not available.")
    ONNX_AVAILABLE = False

# Global CUDA context for thread sharing
_global_cuda_context = None
_cuda_initialized = False

def initialize_cuda_context():
    """Initialize global CUDA context for thread sharing"""
    global _global_cuda_context, _cuda_initialized
    if not _cuda_initialized and TENSORRT_AVAILABLE:
        try:
            cuda.init()
            device = cuda.Device(0)
            _global_cuda_context = device.make_context()
            _global_cuda_context.pop()  # Pop immediately so threads can push/pop
            _cuda_initialized = True
            print("Global CUDA context initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize CUDA context: {e}")
            _cuda_initialized = False

class TensorRTFacialExpressionRecognizer:
    """
    Fixed TensorRT facial expression recognizer for TensorRT 10.3.0
    Updated for PAtt-Lite model with proper preprocessing
    Addresses memory allocation and API compatibility issues
    """
    
    def __init__(self, tensorrt_engine_path=None, onnx_model_path=None, 
                 emotion_labels=None, face_detector_path=None,
                 enable_caching=True, max_cache_size=10):
        
        # Initialize CUDA context if not already done
        initialize_cuda_context()
        
        # PAtt-Lite default emotion labels
        self.emotion_labels = emotion_labels or ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        
        # PAtt-Lite specific constants (precompute for efficiency)
        self.patt_lite_input_size = (224, 224)
        self.patt_lite_mean = np.array([0.4873, 0.4873, 0.4873], dtype=np.float32)
        self.patt_lite_std = np.array([0.2593, 0.2593, 0.2593], dtype=np.float32)
        self.input_height = 224
        self.input_width = 224
        self.num_channels = 3
        self.is_chw = True  # PAtt-Lite uses CHW format
        
        # Preallocate arrays for preprocessing (avoid repeated allocation)
        self.gray_buffer = np.empty((224, 224), dtype=np.uint8)
        self.rgb_buffer = np.empty((224, 224, 3), dtype=np.float32)
        self.normalized_buffer = np.empty((3, 224, 224), dtype=np.float32)
        
        # Initialize backends
        self.use_tensorrt = False
        self.use_onnx = False
        
        # Try TensorRT first
        if TENSORRT_AVAILABLE and tensorrt_engine_path and os.path.exists(tensorrt_engine_path):
            try:
                print(f"Loading TensorRT engine: {tensorrt_engine_path}")
                self._load_tensorrt_engine(tensorrt_engine_path)
                self.use_tensorrt = True
                print("TensorRT engine loaded successfully")
            except Exception as e:
                print(f"Failed to load TensorRT engine: {e}")
                self.use_tensorrt = False
        
        # Fallback to ONNX
        if not self.use_tensorrt and ONNX_AVAILABLE and onnx_model_path and os.path.exists(onnx_model_path):
            try:
                print("Falling back to ONNX Runtime")
                self._load_onnx_model(onnx_model_path)
                self.use_onnx = True
                print("ONNX Runtime fallback loaded")
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                raise RuntimeError("Both TensorRT and ONNX loading failed")
        
        if not (self.use_tensorrt or self.use_onnx):
            raise RuntimeError("No valid inference backend available")
        
        # Initialize face detector
        if face_detector_path is None:
            face_detector_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(face_detector_path)
        if self.face_cascade.empty():
            raise IOError(f"Failed to load face cascade from {face_detector_path}")
        
        # Optimization features (conservative settings for Jetson)
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self.preprocess_cache = {}
        self.cache_lock = Lock()
        
        # Performance monitoring
        self.perf_stats = defaultdict(list)
        self.enable_profiling = False
        
        print(f"FER initialized (TensorRT: {self.use_tensorrt}, ONNX: {self.use_onnx})")
    
    def _load_tensorrt_engine(self, engine_path):
        """Load TensorRT engine with conservative memory allocation for TensorRT 10.3.0"""
        # Push context for engine loading
        if _global_cuda_context:
            _global_cuda_context.push()
        
        try:
            # Initialize TensorRT components
            self.trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
            
            self.context = self.engine.create_execution_context()
            
            # Get tensor information using TensorRT 10.x API
            self.input_tensors = {}
            self.output_tensors = {}
            
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = self.engine.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
                mode = self.engine.get_tensor_mode(name)
                
                # Convert shape to list safely
                try:
                    shape_list = [int(dim) for dim in shape]
                except (TypeError, ValueError):
                    # Handle different TensorRT shape object types
                    shape_list = []
                    try:
                        for j in range(len(shape)):
                            shape_list.append(int(shape[j]))
                    except:
                        # Ultimate fallback - assume PAtt-Lite shape
                        if mode == trt.TensorIOMode.INPUT:
                            shape_list = [-1, 3, 224, 224]  # PAtt-Lite input shape
                        else:
                            shape_list = [-1, len(self.emotion_labels)]
                
                tensor_info = {
                    'name': name,
                    'shape': shape_list,
                    'dtype': dtype,
                    'mode': mode
                }
                
                if mode == trt.TensorIOMode.INPUT:
                    self.input_tensors[name] = tensor_info
                    print(f"Input: {name}, shape: {shape_list}, dtype: {dtype}")
                else:
                    self.output_tensors[name] = tensor_info
                    print(f"Output: {name}, shape: {shape_list}, dtype: {dtype}")
            
            print(f"Model input: {self.input_width}x{self.input_height}x{self.num_channels}, CHW: {self.is_chw}")
            
            # Conservative GPU memory allocation for Jetson
            self.stream = cuda.Stream()
            self._allocate_gpu_memory_conservative()
            
            print("TensorRT engine and GPU memory initialized")
            
        finally:
            # Pop context after engine loading
            if _global_cuda_context:
                _global_cuda_context.pop()

    def _allocate_gpu_memory_conservative(self):
        """Conservative GPU memory allocation for Jetson devices"""
        self.device_inputs = {}
        self.device_outputs = {}
        self.host_inputs = {}
        self.host_outputs = {}
        
        # Calculate actual sizes based on fixed batch size 1
        for name, info in self.input_tensors.items():
            shape = info['shape']
            dtype = info['dtype']
            
            # Use fixed batch size 1 for memory allocation
            actual_shape = [1] + shape[1:] if shape[0] == -1 else shape
            
            # Convert TensorRT dtype to numpy
            if dtype == trt.float32:
                np_dtype = np.float32
                dtype_size = 4
            elif dtype == trt.float16:
                np_dtype = np.float16  
                dtype_size = 2
            elif dtype == trt.int32:
                np_dtype = np.int32
                dtype_size = 4
            else:
                np_dtype = np.float32
                dtype_size = 4
            
            # Calculate size
            size = int(np.prod(actual_shape))
            
            # Use regular numpy arrays for host memory (not pinned) to save memory
            host_mem = np.empty(size, dtype=np_dtype)
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(size * dtype_size)
            
            self.host_inputs[name] = host_mem
            self.device_inputs[name] = device_mem
            
            print(f"Allocated input {name}: {actual_shape} ({size * dtype_size // 1024}KB)")
        
        # Allocate output buffers
        for name, info in self.output_tensors.items():
            shape = info['shape']
            dtype = info['dtype']
            
            actual_shape = [1] + shape[1:] if shape[0] == -1 else shape
            
            if dtype == trt.float32:
                np_dtype = np.float32
                dtype_size = 4
            elif dtype == trt.float16:
                np_dtype = np.float16
                dtype_size = 2
            else:
                np_dtype = np.float32
                dtype_size = 4
            
            size = int(np.prod(actual_shape))
            
            host_mem = np.empty(size, dtype=np_dtype)
            device_mem = cuda.mem_alloc(size * dtype_size)
            
            self.host_outputs[name] = host_mem
            self.device_outputs[name] = device_mem
            
            print(f"Allocated output {name}: {actual_shape} ({size * dtype_size // 1024}KB)")

    def _load_onnx_model(self, onnx_model_path):
        """Load ONNX model as fallback"""
        session_options = onnxruntime.SessionOptions()
        session_options.inter_op_num_threads = 2
        session_options.intra_op_num_threads = 2
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(
            onnx_model_path, 
            providers=providers,
            sess_options=session_options
        )
        
        self.input_name = self.ort_session.get_inputs()[0].name
        model_input_shape = self.ort_session.get_inputs()[0].shape
        
        print(f"ONNX model input: {self.input_width}x{self.input_height}x{self.num_channels}, CHW: {self.is_chw}")
    
    def _preprocess_face_optimized(self, face_roi):
        """Jetson-optimized preprocessing for PAtt-Lite model"""
        start_time = time.time() if self.enable_profiling else None
        
        if face_roi.size == 0:
            return None
        
        try:
            # Step 1: Convert to grayscale (PAtt-Lite was trained on grayscale data)
            if len(face_roi.shape) == 3:
                # Use OpenCV's optimized BGR->GRAY conversion
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            # Step 2: Resize to 224x224 (PAtt-Lite input size)
            if gray_face.shape[:2] != self.patt_lite_input_size:
                cv2.resize(gray_face, self.patt_lite_input_size, 
                          dst=self.gray_buffer, interpolation=cv2.INTER_AREA)
            else:
                self.gray_buffer[:] = gray_face
            
            # Step 3: Convert grayscale to RGB efficiently (3 identical channels)
            # Use broadcasting instead of cv2.cvtColor for better performance
            gray_normalized = self.gray_buffer.astype(np.float32) / 255.0
            
            # Create RGB channels (vectorized operation)
            self.rgb_buffer[:, :, 0] = gray_normalized
            self.rgb_buffer[:, :, 1] = gray_normalized  
            self.rgb_buffer[:, :, 2] = gray_normalized
            
            # Step 4: Apply PAtt-Lite normalization (vectorized)
            self.rgb_buffer = (self.rgb_buffer - self.patt_lite_mean) / self.patt_lite_std
            
            # Step 5: Transpose to CHW format (optimized)
            for c in range(3):
                self.normalized_buffer[c] = self.rgb_buffer[:, :, c]
            
            # Step 6: Add batch dimension
            result = np.expand_dims(self.normalized_buffer, axis=0)
            
            if self.enable_profiling:
                self.perf_stats['preprocess_times'].append(time.time() - start_time)
            
            return result
            
        except Exception as e:
            print(f"Error in PAtt-Lite preprocessing: {e}")
            return None
    
    def _inference_tensorrt(self, input_tensor):
        """Perform inference using TensorRT 10.3.0 API with thread-safe context management"""
        if not _global_cuda_context:
            print("No CUDA context available")
            return None
            
        # Push context for this thread
        _global_cuda_context.push()
        
        try:
            input_name = list(self.input_tensors.keys())[0]
            output_name = list(self.output_tensors.keys())[0]
            
            # Set input shape for dynamic shapes
            input_shape = input_tensor.shape
            if self.input_tensors[input_name]['shape'][0] == -1:
                self.context.set_input_shape(input_name, input_shape)
                
                # Ensure all shapes are specified
                if not self.context.all_binding_shapes_specified:
                    print("Warning: Not all binding shapes specified")
            
            # Copy input data to host buffer
            np.copyto(self.host_inputs[input_name], input_tensor.ravel())
            
            # Transfer to GPU
            cuda.memcpy_htod_async(self.device_inputs[input_name], self.host_inputs[input_name], self.stream)
            
            # Set tensor addresses for TensorRT 10.x
            self.context.set_tensor_address(input_name, int(self.device_inputs[input_name]))
            self.context.set_tensor_address(output_name, int(self.device_outputs[output_name]))
            
            # Execute inference using TensorRT 10.3.0 API
            success = self.context.execute_async_v3(self.stream.handle)
            
            if not success:
                print("TensorRT execution failed")
                return None
            
            # Transfer result back
            cuda.memcpy_dtoh_async(self.host_outputs[output_name], self.device_outputs[output_name], self.stream)
            self.stream.synchronize()
            
            # Reshape output
            output_shape = self.output_tensors[output_name]['shape']
            if output_shape[0] == -1:
                actual_output_shape = [1] + output_shape[1:]
            else:
                actual_output_shape = output_shape
            
            result = self.host_outputs[output_name].reshape(actual_output_shape)
            return result[0]  # Remove batch dimension
            
        except Exception as e:
            print(f"TensorRT inference error: {e}")
            return None
        finally:
            # Always pop context
            _global_cuda_context.pop()
    
    def _inference_onnx(self, input_tensor):
        """Perform inference using ONNX Runtime"""
        try:
            ort_inputs = {self.input_name: input_tensor}
            ort_outs = self.ort_session.run(None, ort_inputs)
            return ort_outs[0][0]  # Remove batch dimension
        except Exception as e:
            print(f"ONNX inference error: {e}")
            return None
    
    def detect_and_recognize_optimized(self, frame):
        """Optimized detection and recognition pipeline"""
        if frame is None:
            return []
        
        # Face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_coords = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),  # Larger minimum size for better detection
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detected_expressions = []
        
        for (x, y, w, h) in faces_coords:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
            
            # Preprocess
            input_tensor = self._preprocess_face_optimized(face_roi)
            if input_tensor is None:
                continue
            
            # Inference
            start_inference = time.time() if self.enable_profiling else None
            
            if self.use_tensorrt:
                emotion_logits = self._inference_tensorrt(input_tensor)
            else:
                emotion_logits = self._inference_onnx(input_tensor)
            
            if emotion_logits is None:
                continue
            
            if self.enable_profiling:
                self.perf_stats['inference_times'].append(time.time() - start_inference)
            
            # Get prediction
            predicted_index = np.argmax(emotion_logits)
            
            if 0 <= predicted_index < len(self.emotion_labels):
                emotion_text = self.emotion_labels[predicted_index]
            else:
                emotion_text = "Unknown"
            
            detected_expressions.append({
                "box": (x, y, w, h),
                "emotion": emotion_text,
                "emotion_index": predicted_index
            })
        
        return detected_expressions
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {}
        for key, times in self.perf_stats.items():
            if times:
                stats[key] = {
                    'avg': np.mean(times) * 1000,
                    'min': np.min(times) * 1000,
                    'max': np.max(times) * 1000,
                    'count': len(times)
                }
        
        stats['backend'] = 'TensorRT' if self.use_tensorrt else 'ONNX'
        stats['input_size'] = f"{self.input_width}x{self.input_height}"
        
        return stats
    
    def enable_performance_profiling(self, enable=True):
        """Enable/disable performance profiling"""
        self.enable_profiling = enable
        if not enable:
            self.perf_stats.clear()
    
    def cleanup(self):
        """Cleanup GPU resources"""
        if self.use_tensorrt and _global_cuda_context:
            _global_cuda_context.push()
            try:
                for mem in self.device_inputs.values():
                    mem.free()
                for mem in self.device_outputs.values():
                    mem.free()
                print("GPU memory cleaned up")
            except Exception as e:
                print(f"Warning: GPU cleanup error: {e}")
            finally:
                _global_cuda_context.pop()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
    
    # Backward compatibility
    def detect_and_recognize(self, frame):
        """Backward compatible method"""
        return self.detect_and_recognize_optimized(frame)

if __name__ == '__main__':
    # Test the fixed TensorRT FER module with PAtt-Lite
    TRT_ENGINE_PATH = "models/patt_lite_enhanced.trt"
    ONNX_MODEL_PATH = "models/patt_lite_enhanced.onnx"
    EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
    
    print("Testing Fixed TensorRT FER Module with PAtt-Lite")
    print("=" * 50)
    
    try:
        # Initialize recognizer
        recognizer = TensorRTFacialExpressionRecognizer(
            tensorrt_engine_path=TRT_ENGINE_PATH,
            onnx_model_path=ONNX_MODEL_PATH,
            emotion_labels=EMOTIONS,
            enable_caching=True,
            max_cache_size=5  # Conservative cache size
        )
        recognizer.enable_performance_profiling(True)
        
        print(f"\nRecognizer Info:")
        stats = recognizer.get_performance_stats()
        print(f"  Backend: {stats['backend']}")
        print(f"  Input size: {stats.get('input_size', 'Unknown')}")
        
        # Test with camera if available
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print(f"\nTesting with camera (press 'q' to quit)...")
            
            frame_count = 0
            while frame_count < 50:  # Test for 50 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                expressions = recognizer.detect_and_recognize_optimized(frame)
                
                # Draw results
                for expr in expressions:
                    x, y, w, h = expr["box"]
                    emotion = expr["emotion"]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Fixed TensorRT FER Test - PAtt-Lite", frame)
                
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final stats
            final_stats = recognizer.get_performance_stats()
            print(f"\n=== PERFORMANCE RESULTS ===")
            for key, metrics in final_stats.items():
                if isinstance(metrics, dict) and 'avg' in metrics:
                    print(f"{key}: {metrics['avg']:.2f}ms avg (count: {metrics['count']})")
                else:
                    print(f"{key}: {metrics}")
        
        else:
            print("Camera not available - creating test image")
            # Create a test image
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            expressions = recognizer.detect_and_recognize_optimized(test_image)
            print(f"Test completed: {len(expressions)} faces detected")
        
        # Cleanup
        recognizer.cleanup()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
