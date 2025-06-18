# cgan_tensorrt_module_v2_fixed.py
"""
TensorRT module for TensorRT 10.3.0 with smooth video-like animation
"""
import numpy as np
import cv2
import os
import time
from collections import deque

# Check TensorRT availability
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # DO NOT import pycuda.autoinit - this causes threading issues
    TENSORRT_AVAILABLE = True
    print(f"TensorRT version: {trt.__version__}")
except ImportError:
    print("Warning: TensorRT or PyCUDA not available.")
    TENSORRT_AVAILABLE = False

# Check TensorFlow availability  
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available.")
    TF_AVAILABLE = False

# Import fallback only if TensorFlow is available
if TF_AVAILABLE:
    try:
        from cgan_module import CGANAnimator as FallbackCGANAnimator
        FALLBACK_AVAILABLE = True
    except ImportError:
        print("Warning: cgan_module fallback not available.")
        FALLBACK_AVAILABLE = False
else:
    FALLBACK_AVAILABLE = False

# Configuration
IMG_SIZE = 256
CHANNELS = 3
NUM_EMOTIONS = 7
LATENT_DIM = 128

# Enhanced animation parameters for smooth video-like transitions
SMOOTH_INTERPOLATION_STEPS = 90  # Increased from 15 for much smoother transitions
TEMPORAL_SMOOTHING_WINDOW = 15   # Frames to smooth over
TRANSITION_BLEND_FRAMES = 30     # Frames for emotion transitions

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
            print("Global CUDA context initialized for cGAN")
        except Exception as e:
            print(f"Warning: Failed to initialize CUDA context for cGAN: {e}")
            _cuda_initialized = False

class TemporalSmoother:
    """Temporal smoothing for latent vectors to create video-like continuity"""
    
    def __init__(self, window_size=TEMPORAL_SMOOTHING_WINDOW, latent_dim=LATENT_DIM):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.history = deque(maxlen=window_size)
        self.weights = self._compute_smoothing_weights()
    
    def _compute_smoothing_weights(self):
        """Compute temporal smoothing weights (more recent frames have higher weight)"""
        weights = np.exp(np.linspace(-2, 0, self.window_size))
        return weights / np.sum(weights)
    
    def smooth(self, new_latent):
        """Apply temporal smoothing to reduce flickering"""
        self.history.append(new_latent.copy())
        
        if len(self.history) < 2:
            return new_latent
        
        # Weighted average with more recent frames having higher influence
        smoothed = np.zeros_like(new_latent)
        total_weight = 0
        
        for i, latent in enumerate(self.history):
            weight = self.weights[i] if i < len(self.weights) else self.weights[-1]
            smoothed += weight * latent
            total_weight += weight
        
        return smoothed / total_weight if total_weight > 0 else new_latent

class TensorRTCGANAnimator:
    """
    Enhanced cGAN animator with smooth video-like transitions
    Conservative memory allocation for Jetson devices
    """
    
    def __init__(self, tensorrt_engine_path=None, tensorflow_weights_path=None, 
                 latent_dim=LATENT_DIM, num_emotions=NUM_EMOTIONS, 
                 img_size=IMG_SIZE, enable_smooth_animation=True):
        
        # Initialize CUDA context if not already done
        initialize_cuda_context()
        
        self.latent_dim = latent_dim
        self.num_emotions = num_emotions
        self.img_size = img_size
        self.enable_smooth_animation = enable_smooth_animation
        
        # Initialize backends
        self.use_tensorrt = False
        self.fallback_animator = None
        
        # Try TensorRT first
        if TENSORRT_AVAILABLE and tensorrt_engine_path and os.path.exists(tensorrt_engine_path):
            try:
                print(f"Loading TensorRT engine from {tensorrt_engine_path}")
                self._load_tensorrt_engine(tensorrt_engine_path)
                self.use_tensorrt = True
                print("TensorRT engine loaded successfully")
            except Exception as e:
                print(f"Failed to load TensorRT engine: {e}")
                self.use_tensorrt = False
        
        # Fallback to TensorFlow
        if not self.use_tensorrt:
            if FALLBACK_AVAILABLE and tensorflow_weights_path and os.path.exists(tensorflow_weights_path):
                print("Falling back to TensorFlow implementation")
                self.fallback_animator = FallbackCGANAnimator(tensorflow_weights_path)
                print("TensorFlow fallback loaded")
            else:
                raise ValueError("Either valid TensorRT engine or TensorFlow weights required")
        
        # Enhanced animation state for smooth transitions
        self._init_enhanced_animation_state()
        
        print(f"Enhanced cGAN Animator initialized (TensorRT: {self.use_tensorrt}, Smooth: {enable_smooth_animation})")

    def _init_enhanced_animation_state(self):
        """Initialize enhanced animation state for smooth transitions"""
        # Multiple latent vectors for complex interpolation
        self.current_latent_z = np.random.normal(0, 1, (1, self.latent_dim)).astype(np.float32)
        self.target_latent_z = np.random.normal(0, 1, (1, self.latent_dim)).astype(np.float32)
        self.next_target_z = np.random.normal(0, 1, (1, self.latent_dim)).astype(np.float32)
        
        # Enhanced interpolation parameters
        self.interpolation_alpha = 0.0
        self.interpolation_total_steps = SMOOTH_INTERPOLATION_STEPS
        self.last_emotion_index = -1
        
        # Temporal smoothing
        self.temporal_smoother = TemporalSmoother() if self.enable_smooth_animation else None
        
        # Transition management
        self.in_transition = False
        self.transition_progress = 0.0
        self.transition_source_z = None
        self.transition_target_z = None
        
        # Timing for frame-rate independent animation
        self.last_generation_time = time.time()
        self.target_fps = 30  # Target animation FPS
        self.frame_duration = 1.0 / self.target_fps

    def _load_tensorrt_engine(self, engine_path):
        """Load TensorRT engine with TensorRT 10.3.0 API and conservative memory allocation"""
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
                    # Handle different shape object types
                    shape_list = []
                    try:
                        for j in range(len(shape)):
                            shape_list.append(int(shape[j]))
                    except:
                        # Ultimate fallback for cGAN shapes
                        if mode == trt.TensorIOMode.INPUT:
                            if 'noise' in name.lower() or len(name) > 5:
                                shape_list = [1, self.latent_dim]
                            else:
                                shape_list = [1, 1]
                        else:
                            shape_list = [1, 3, self.img_size, self.img_size]
                
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
            
            # Identify input tensors
            self.noise_input_name = None
            self.label_input_name = None
            
            for name, info in self.input_tensors.items():
                shape = info['shape']
                if len(shape) >= 2 and shape[1] == self.latent_dim:
                    self.noise_input_name = name
                elif len(shape) >= 2 and shape[1] == 1:
                    self.label_input_name = name
                elif 'noise' in name.lower():
                    self.noise_input_name = name
                elif 'label' in name.lower():
                    self.label_input_name = name
            
            # Fallback assignment if not found
            input_names = list(self.input_tensors.keys())
            if self.noise_input_name is None:
                self.noise_input_name = input_names[0] if input_names else None
            if self.label_input_name is None and len(input_names) > 1:
                self.label_input_name = input_names[1]
            
            print(f"Mapped inputs: noise='{self.noise_input_name}', label='{self.label_input_name}'")
            
            # Get output tensor info
            self.output_name = list(self.output_tensors.keys())[0]
            self.output_shape = self.output_tensors[self.output_name]['shape']
            
            # Conservative GPU memory allocation
            self.stream = cuda.Stream()
            self._allocate_gpu_memory_conservative()
            
            print("TensorRT engine and GPU memory initialized")
            
        finally:
            # Pop context after engine loading
            if _global_cuda_context:
                _global_cuda_context.pop()

    def _allocate_gpu_memory_conservative(self):
        """Conservative GPU memory allocation for Jetson devices"""
        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}
        
        # Allocate input buffers
        for name, info in self.input_tensors.items():
            shape = info['shape']
            dtype = info['dtype']
            
            # Use fixed batch size 1
            actual_shape = [1] + shape[1:] if shape[0] == -1 else shape
            
            # Convert dtype
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
            
            size = int(np.prod(actual_shape))
            
            # Use regular numpy arrays (not pinned) to conserve memory
            host_mem = np.empty(size, dtype=np_dtype)
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

    def _smooth_interpolation(self, alpha, source_z, target_z):
        """Enhanced interpolation for smoother transitions"""
        if not self.enable_smooth_animation:
            # Linear interpolation for fallback
            return (1.0 - alpha) * source_z + alpha * target_z
        
        # Use cosine interpolation for smoother curves (better than SLERP for video-like motion)
        smooth_alpha = 0.5 * (1.0 - np.cos(alpha * np.pi))
        
        # Apply interpolation
        interpolated = (1.0 - smooth_alpha) * source_z + smooth_alpha * target_z
        
        # Apply temporal smoothing to reduce flicker
        if self.temporal_smoother:
            interpolated = self.temporal_smoother.smooth(interpolated)
        
        return interpolated

    def _update_animation_timing(self):
        """Frame-rate independent animation timing"""
        current_time = time.time()
        delta_time = current_time - self.last_generation_time
        self.last_generation_time = current_time
        
        # Calculate interpolation speed based on target FPS
        time_step = delta_time / self.frame_duration
        return min(time_step, 2.0)  # Cap to prevent large jumps

    def set_smooth_animation(self, enabled):
        """Enable or disable smooth animation features"""
        self.enable_smooth_animation = enabled
        if enabled and not self.temporal_smoother:
            self.temporal_smoother = TemporalSmoother()
        print(f"Smooth animation: {'enabled' if enabled else 'disabled'}")

    def generate_image(self, emotion_index, use_slerp=None):
        """Generate image with enhanced smooth interpolation"""
        if not (0 <= emotion_index < self.num_emotions):
            print(f"Warning: Invalid emotion_index {emotion_index}. Defaulting to 0.")
            emotion_index = 0
        
        # Frame-rate independent timing
        time_step = self._update_animation_timing()
        
        # Handle emotion changes with smooth transitions
        if self.last_emotion_index != emotion_index:
            self._start_emotion_transition(emotion_index)
            self.last_emotion_index = emotion_index

        # Update interpolation state
        if self.in_transition:
            self._update_transition(time_step)
        else:
            self._update_normal_interpolation(time_step)
        
        # Get current interpolated latent vector
        interpolated_z = self._get_current_interpolated_latent()
        
        # Generate image
        if self.use_tensorrt:
            return self._generate_image_tensorrt(interpolated_z.astype(np.float32), emotion_index)
        else:
            return self._generate_image_fallback(interpolated_z, emotion_index)

    def _start_emotion_transition(self, new_emotion_index):
        """Start smooth transition to new emotion"""
        # Store current state as transition source
        self.transition_source_z = self._get_current_interpolated_latent().copy()
        
        # Generate new target for the emotion
        self.transition_target_z = np.random.normal(0, 1, (1, self.latent_dim)).astype(np.float32)
        
        # Start transition
        self.in_transition = True
        self.transition_progress = 0.0

    def _update_transition(self, time_step):
        """Update emotion transition state"""
        # Progress transition
        transition_speed = 1.0 / TRANSITION_BLEND_FRAMES
        self.transition_progress += transition_speed * time_step
        
        if self.transition_progress >= 1.0:
            # Transition complete
            self.current_latent_z = self.transition_target_z.copy()
            self.target_latent_z = np.random.normal(0, 1, (1, self.latent_dim)).astype(np.float32)
            self.interpolation_alpha = 0.0
            self.in_transition = False

    def _update_normal_interpolation(self, time_step):
        """Update normal interpolation state"""
        # Normal interpolation progress
        interpolation_speed = 1.0 / self.interpolation_total_steps
        self.interpolation_alpha += interpolation_speed * time_step
        
        if self.interpolation_alpha >= 1.0:
            # Move to next interpolation cycle
            self.current_latent_z = self.target_latent_z.copy()
            self.target_latent_z = self.next_target_z.copy()
            self.next_target_z = np.random.normal(0, 1, (1, self.latent_dim)).astype(np.float32)
            self.interpolation_alpha = 0.0

    def _get_current_interpolated_latent(self):
        """Get current interpolated latent vector"""
        if self.in_transition:
            # Smooth transition between emotions
            return self._smooth_interpolation(
                self.transition_progress, 
                self.transition_source_z, 
                self.transition_target_z
            )
        else:
            # Normal interpolation
            return self._smooth_interpolation(
                self.interpolation_alpha, 
                self.current_latent_z, 
                self.target_latent_z
            )

    def _generate_image_tensorrt(self, interpolated_z, emotion_index):
        """Generate image using TensorRT 10.3.0 API with thread-safe context management"""
        if not _global_cuda_context:
            print("No CUDA context available")
            return self._create_error_image("No CUDA Context")
            
        # Push context for this thread
        _global_cuda_context.push()
        
        try:
            # Prepare inputs
            noise = interpolated_z.astype(np.float32)
            label = np.array([[emotion_index]], dtype=np.int32)
            
            # Copy inputs to host buffers
            if self.noise_input_name:
                np.copyto(self.host_inputs[self.noise_input_name], noise.ravel())
            
            if self.label_input_name:
                np.copyto(self.host_inputs[self.label_input_name], label.ravel())
            
            # Set input shapes for dynamic shapes
            for name, info in self.input_tensors.items():
                if info['shape'][0] == -1:  # Dynamic batch
                    if name == self.noise_input_name:
                        self.context.set_input_shape(name, (1, self.latent_dim))
                    elif name == self.label_input_name:
                        self.context.set_input_shape(name, (1, 1))
            
            # Transfer inputs to GPU
            for name in self.input_tensors.keys():
                cuda.memcpy_htod_async(self.device_inputs[name], self.host_inputs[name], self.stream)
            
            # Set tensor addresses for TensorRT 10.x
            for name in self.input_tensors.keys():
                self.context.set_tensor_address(name, int(self.device_inputs[name]))
            
            for name in self.output_tensors.keys():
                self.context.set_tensor_address(name, int(self.device_outputs[name]))
            
            # Execute inference using TensorRT 10.3.0 API
            success = self.context.execute_async_v3(self.stream.handle)
            
            if not success:
                print("TensorRT execution failed")
                return self._create_error_image("TensorRT Exec Failed")
            
            # Transfer outputs back to CPU
            for name in self.output_tensors.keys():
                cuda.memcpy_dtoh_async(self.host_outputs[name], self.device_outputs[name], self.stream)
            
            # Synchronize
            self.stream.synchronize()
            
            # Process output
            output_data = self.host_outputs[self.output_name]
            
            # Determine actual output shape
            output_shape = self.output_shape
            if output_shape[0] == -1:
                actual_output_shape = [1] + output_shape[1:]
            else:
                actual_output_shape = output_shape
            
            # Reshape to image format
            if len(actual_output_shape) == 4:  # NCHW or NHWC
                if actual_output_shape[1] == CHANNELS:  # NCHW format
                    image = output_data.reshape(actual_output_shape).transpose(0, 2, 3, 1)[0]  # Convert to HWC
                else:  # NHWC format
                    image = output_data.reshape(actual_output_shape)[0]
            else:
                # Fallback reshape
                image = output_data.reshape((self.img_size, self.img_size, CHANNELS))
            
            # Convert from [-1, 1] to [0, 255]
            image = np.clip((image + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = image
            
            return bgr_image
            
        except Exception as e:
            print(f"TensorRT inference error: {e}")
            return self._create_error_image("TensorRT Error")
        finally:
            # Always pop context
            _global_cuda_context.pop()

    def _create_error_image(self, error_text):
        """Create an error placeholder image"""
        placeholder = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        cv2.putText(placeholder, error_text, (50, self.img_size//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return placeholder

    def _generate_image_fallback(self, interpolated_z, emotion_index):
        """Generate image using TensorFlow fallback"""
        if self.fallback_animator is None:
            return self._create_error_image("No Backend")
        
        try:
            # Synchronize interpolation state with fallback
            if hasattr(self.fallback_animator, 'current_latent_z') and TF_AVAILABLE:
                self.fallback_animator.current_latent_z = tf.constant(interpolated_z)
                self.fallback_animator.interpolation_alpha = 0.0
            
            return self.fallback_animator.generate_image(emotion_index, use_slerp=False)
        except Exception as e:
            print(f"Fallback inference error: {e}")
            return self._create_error_image("Fallback Error")

    def get_performance_info(self):
        """Get performance information"""
        info = {
            'backend': 'TensorRT' if self.use_tensorrt else 'TensorFlow',
            'model_size': f"{self.img_size}x{self.img_size}",
            'num_emotions': self.num_emotions,
            'latent_dim': self.latent_dim,
            'smooth_animation': self.enable_smooth_animation,
            'interpolation_steps': self.interpolation_total_steps,
            'transition_frames': TRANSITION_BLEND_FRAMES
        }
        
        if self.use_tensorrt:
            info.update({
                'num_io_tensors': self.engine.num_io_tensors,
                'input_tensors': list(self.input_tensors.keys()),
                'output_tensors': list(self.output_tensors.keys())
            })
        
        return info

    def get_animation_status(self):
        """Get current animation status for debugging"""
        return {
            'interpolation_alpha': f"{self.interpolation_alpha:.3f}",
            'in_transition': self.in_transition,
            'transition_progress': f"{self.transition_progress:.3f}" if self.in_transition else "N/A",
            'last_emotion': self.last_emotion_index,
            'temporal_smoothing': self.temporal_smoother is not None
        }

    def cleanup(self):
        """Cleanup GPU resources"""
        if self.use_tensorrt and _global_cuda_context:
            _global_cuda_context.push()
            try:
                for mem in self.device_inputs.values():
                    mem.free()
                for mem in self.device_outputs.values():
                    mem.free()
                print("cGAN GPU memory cleaned up")
            except Exception as e:
                print(f"Warning: cGAN GPU cleanup error: {e}")
            finally:
                _global_cuda_context.pop()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def benchmark_performance(animator, num_iterations=30):
    """Benchmark the performance of the enhanced animator"""
    print(f"Benchmarking enhanced cGAN performance with {num_iterations} iterations...")
    
    # Warm up
    for i in range(3):
        _ = animator.generate_image(i % NUM_EMOTIONS)
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        emotion_idx = i % NUM_EMOTIONS
        image = animator.generate_image(emotion_idx)
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        if i % 10 == 0:
            print(f"Iteration {i}: {inference_time*1000:.2f}ms, image shape: {image.shape}")
            
            # Print animation status
            status = animator.get_animation_status()
            print(f"  Animation: alpha={status['interpolation_alpha']}, transition={status['in_transition']}")
    
    # Calculate statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)
    fps = 1.0 / avg_time
    
    print(f"\nEnhanced cGAN Performance Results:")
    print(f"Average inference time: {avg_time*1000:.2f}ms (+/-{std_time*1000:.2f}ms)")
    print(f"Min/Max time: {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
    print(f"Throughput: {fps:.1f} FPS")
    print(f"Backend: {animator.get_performance_info()['backend']}")
    print(f"Smooth animation: {animator.enable_smooth_animation}")
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': fps,
        'backend': animator.get_performance_info()['backend'],
        'smooth_enabled': animator.enable_smooth_animation
    }

if __name__ == '__main__':
    # Test the enhanced TensorRT cGAN module
    TRT_ENGINE_PATH = "models/cgan_generator.trt"
    TF_WEIGHTS_PATH = "models/best_generator.weights.h5"
    
    print("Testing Enhanced TensorRT cGAN Module with Smooth Animation")
    print("=" * 50)
    
    try:
        # Initialize enhanced animator
        animator = TensorRTCGANAnimator(
            tensorrt_engine_path=TRT_ENGINE_PATH,
            tensorflow_weights_path=TF_WEIGHTS_PATH,
            enable_smooth_animation=True
        )
        
        print(f"\nEnhanced Animator Info:")
        info = animator.get_performance_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test smooth animation
        print(f"\nTesting smooth animation transitions...")
        emotion_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
        
        # Test emotion transitions
        for i in range(len(emotion_names)):
            emotion_name = emotion_names[i]
            print(f"\nTransitioning to: {emotion_name}")
            
            # Generate multiple frames for this emotion to see smooth transitions
            for frame in range(20):
                start_time = time.time()
                image = animator.generate_image(i)
                inference_time = time.time() - start_time
                
                if frame % 10 == 0:
                    status = animator.get_animation_status()
                    print(f"  Frame {frame}: {inference_time*1000:.2f}ms, alpha={status['interpolation_alpha']}")
                
                # Save sample frames
                if frame == 10:
                    output_path = f"models/test_smooth_{emotion_name}.jpg"
                    success = cv2.imwrite(output_path, image)
                    if success:
                        print(f"    Saved: {output_path}")
        
        # Performance benchmark
        print(f"\nRunning enhanced performance benchmark...")
        benchmark_results = benchmark_performance(animator, num_iterations=30)
        
        # Test smooth vs non-smooth
        print(f"\nTesting smooth animation toggle...")
        animator.set_smooth_animation(False)
        quick_test = benchmark_performance(animator, num_iterations=10)
        
        animator.set_smooth_animation(True)
        smooth_test = benchmark_performance(animator, num_iterations=10)
        
        print(f"\nComparison:")
        print(f"  Non-smooth: {quick_test['avg_time_ms']:.2f}ms avg")
        print(f"  Smooth: {smooth_test['avg_time_ms']:.2f}ms avg")
        print(f"  Overhead: {smooth_test['avg_time_ms'] - quick_test['avg_time_ms']:.2f}ms")
        
        # Summary
        if animator.use_tensorrt:
            print(f"\nTensorRT acceleration active with smooth animation!")
            print(f"Video-like continuity: ENABLED")
            print(f"Interpolation steps: {animator.interpolation_total_steps}")
            print(f"Transition frames: {TRANSITION_BLEND_FRAMES}")
        else:
            print(f"\nUsing TensorFlow fallback with smooth animation")
            print(f"For optimal performance, ensure TensorRT engine is available")
        
        # Cleanup
        animator.cleanup()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()