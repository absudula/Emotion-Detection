# cgan_module.py
"""
Handles the cGAN model for generating animated faces based on emotion.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
import numpy as np
import cv2

# --- Configurations (match with the trained cGAN model) ---
IMG_SIZE = 256       # Output image size
CHANNELS = 3         # Number of image channels (RGB)
NUM_EMOTIONS = 7     # Number of emotion classes the cGAN was trained on
LATENT_DIM = 128     # Size of the latent (noise) vector
EMBEDDING_DIM = 64   # Dimension of the emotion embedding

# --- Model Architecture Components ---

class LabelEmbedding(layers.Layer):
    """
    Custom Keras layer for embedding integer labels.
    As defined in the provided PDF.
    """
    def __init__(self, num_classes, embedding_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.embedding = layers.Embedding(num_classes, embedding_dim)

    def call(self, labels):
        return self.embedding(labels)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
        })
        return config

def build_generator(latent_dim=LATENT_DIM, num_emotions=NUM_EMOTIONS, embedding_dim=EMBEDDING_DIM):
    """
    Builds the cGAN Generator model.
    """
    init = initializers.GlorotUniform()

    # Inputs
    noise_input = layers.Input(shape=(latent_dim,), name="noise_input")
    label_input = layers.Input(shape=(1,), dtype='int32', name="label_input")

    # Label embedding
    label_emb = LabelEmbedding(num_emotions, embedding_dim)(label_input) # Output shape: (None, 1, embedding_dim)
    label_emb_flat = layers.Flatten()(label_emb) # Output shape: (None, embedding_dim)

    # Concatenate noise and label embedding
    combined = layers.Concatenate()([noise_input, label_emb_flat]) # Shape: (None, latent_dim + embedding_dim)

    # Initial dense layer adjusted for 256x256 (needs 6 upsampling stages to go from 4x4 to 256x256)
    x = layers.Dense(4 * 4 * 1024, use_bias=False, kernel_initializer=init)(combined)
    x = layers.Reshape((4, 4, 1024))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Upsampling blocks (6 stages: 4->8, 8->16, 16->32, 32->64, 64->128, 128->256)
    
    # Stage 1: 4x4 -> 8x8
    x = layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 2: 8x8 -> 16x16
    x = layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 3: 16x16 -> 32x32
    x = layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 4: 32x32 -> 64x64
    x = layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Stage 5: 64x64 -> 128x128
    x = layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Stage 6: 128x128 -> 256x256
    x = layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final convolutional layer for smoothing
    x = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Output layer (outputs RGB, range [-1, 1])
    output_image = layers.Conv2D(CHANNELS, (3, 3), padding='same', activation='tanh', kernel_initializer=init)(x)

    model = models.Model([noise_input, label_input], output_image, name='generator')
    return model

class CGANAnimator:
    """
    Manages the cGAN model to generate images based on emotions,
    including latent space interpolation for smoother animations.
    """
    def __init__(self, generator_weights_path,
                 latent_dim=LATENT_DIM, num_emotions=NUM_EMOTIONS,
                 embedding_dim=EMBEDDING_DIM, img_size=IMG_SIZE):
        self.latent_dim = latent_dim
        self.num_emotions = num_emotions
        self.img_size = img_size
        
        try:
            self.generator = build_generator(latent_dim, num_emotions, embedding_dim)
            self.generator.load_weights(generator_weights_path)
            print(f"cGAN generator model built and weights loaded from {generator_weights_path}")
        except Exception as e:
            raise RuntimeError(f"Error building or loading cGAN generator: {e}. Ensure TensorFlow is installed and paths are correct.")

        self.current_latent_z = tf.random.normal([1, self.latent_dim])
        self.target_latent_z = tf.random.normal([1, self.latent_dim])
        self.interpolation_alpha = 0.0
        self.interpolation_total_steps = 15 
        self.last_emotion_index = -1

    def _slerp(self, val, low, high):
        omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1.0, 1.0))
        so = np.sin(omega)
        if so == 0:
            return (1.0-val) * low + val * high
        return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

    def generate_image(self, emotion_index, use_slerp=False):
        if not (0 <= emotion_index < self.num_emotions):
            print(f"Warning: Invalid emotion_index {emotion_index}. Defaulting to 0.")
            emotion_index = 0
            
        if self.interpolation_alpha >= 1.0 or self.last_emotion_index != emotion_index:
            self.current_latent_z = self.target_latent_z 
            self.target_latent_z = tf.random.normal([1, self.latent_dim])
            self.interpolation_alpha = 0.0
            self.last_emotion_index = emotion_index

        alpha = self.interpolation_alpha / self.interpolation_total_steps
        if use_slerp:
            z_interp_np = self._slerp(alpha, self.current_latent_z.numpy().flatten(), self.target_latent_z.numpy().flatten())
            interpolated_z = tf.constant(np.expand_dims(z_interp_np, axis=0), dtype=tf.float32)
        else: 
            interpolated_z = (1.0 - alpha) * self.current_latent_z + alpha * self.target_latent_z
        
        self.interpolation_alpha += 1.0

        emotion_label_tensor = tf.constant([[emotion_index]], dtype='int32')
        generated_image_tensor = self.generator([interpolated_z, emotion_label_tensor], training=False)
        
        generated_image_np = (generated_image_tensor[0].numpy() + 1.0) / 2.0 * 255.0
        generated_image_np = np.clip(generated_image_np, 0, 255)
        rgb_image = generated_image_np.astype(np.uint8)
        
        # Convert RGB to BGR for display ---
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return bgr_image # Return the BGR image

if __name__ == '__main__':
    CGAN_WEIGHTS_PATH = "models/best_generator.weights.h5"
    
    import os
    if not os.path.exists(CGAN_WEIGHTS_PATH):
        print(f"Could not find model weights")
        exit()
    try:
        animator = CGANAnimator(CGAN_WEIGHTS_PATH)
        print("cGAN Animator initialized. Generating sample images...")
    except Exception as e:
        print(f"Failed to initialize CGANAnimator: {e}")
        exit()

    current_emotion_idx_test = 0
    emotion_names_test = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']

    cv2.namedWindow("cGAN Test Output", cv2.WINDOW_NORMAL)

    for i in range(200): 
        if i % 30 == 0: 
            current_emotion_idx_test = (current_emotion_idx_test + 1) % NUM_EMOTIONS
            print(f"Generating for emotion: {emotion_names_test[current_emotion_idx_test]}")

        generated_img_bgr = animator.generate_image(current_emotion_idx_test, use_slerp=True) # Now returns BGR

        if generated_img_bgr is not None:
            display_img = cv2.resize(generated_img_bgr, (512, 512)) 
            cv2.putText(display_img, f"{emotion_names_test[current_emotion_idx_test]} (Frame {i})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green text
            cv2.imshow("cGAN Test Output", display_img)
        
        if cv2.waitKey(50) & 0xFF == ord('q'): 
            break
            
    cv2.destroyAllWindows()
    print("cGAN Animator execution finished.")