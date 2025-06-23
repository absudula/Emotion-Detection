# Real-Time Facial Expression Recognition & Response System for Children

A real-time AI system that detects children's facial expressions and generates responsive animated character feedback, designed for educational technology and therapeutic applications. 

## Overview

This project combines advanced facial expression recognition with conditional generative adversarial networks to create an interactive system that responds to children's emotions in real-time. The system detects facial expressions through a camera feed and generates appropriate animated character responses to support emotional development and engagement. A quick overview of the end-to-end system is explained in the [PROJECT_POSTER](./PROJECT_POSTER.pdf) document.

## Key Features

- **Real-time facial expression recognition** with 93.71% accuracy across 7 emotion classes
- **Animated character response generation** using conditional GANs
- **Edge-optimized deployment** on NVIDIA Jetson Orin Nano achieving ~30 FPS
- **Rule-based emotion mapping** implementing therapeutic response strategies
- **TensorRT acceleration** for optimized inference performance

## System Architecture

### Frontend: Facial Expression Recognition (FER)
- **Enhanced PAtt-Lite model** with MobileNetV1 backbone
- **Multi-head attention classifier** for improved expression classification  
- **Two-stage training strategy** for optimal convergence
- **Multi-dataset training** on RAF-DB, CK+, and FER+ datasets

### Backend: Animated Character Generation
- **Conditional GAN (cGAN)** for 256x256 facial expression synthesis
- **6-stage transposed convolution** upsampling architecture
- **Emotion-conditioned generation** across 7 emotion classes
- **Trained on FERG-DB dataset** for consistent character animations

### Rule-Based Emotion Mapping
- **Therapeutic response sequences** based on child psychology principles
- **Validation-first approach** with gradual emotional regulation
- **Dual-mode operation** for therapeutic and mirror responses

## Performance Results

- **FER Accuracy**: 93.71% on combined multi-dataset evaluation
- **Real-time Performance**: ~30 FPS on NVIDIA Jetson Orin Nano
- **Per-class F1-scores**: 0.847 to 0.993 across emotion categories
- **Edge Optimization**: TensorRT 10.3.0 acceleration with FP16 precision

## Key Highlights

- **Lightweight Architecture**: Optimized for edge deployment with minimal computational overhead
- **Multi-threaded Pipeline**: Separate threads for camera capture, inference, and display operations
- **CUDA Context Management**: Thread-safe GPU resource handling for stable performance
- **Memory Optimization**: Conservative allocation strategies for Jetson hardware constraints

## Emotion Classes

The system recognizes and responds to 7 primary emotions:
- Anger
- Disgust  
- Fear
- Happiness
- Sadness
- Surprise
- Neutral

## Hardware Requirements

- **Primary Target**: NVIDIA Jetson Orin Nano
- **GPU Memory**: 8GB recommended
- **Camera**: USB or CSI camera support
- **Software**: TensorRT 10.3.0, CUDA support

## Repository Structure

- [jetson_src](./jetson_src): Contains the end-to-end system inference logic on NVIDIA Jetson Orin Nano board.
- [nn_models_src](./nn_models_src): Contains the python implementation for building, training, and evaluating the Facial Expression Recognition and cGAN models.
- [outputs](./outputs): Contains the sample output results of the system execution on Jetson.
- [doc](./doc): Contains the detailed project report explaining all the key details of the project. Go through [PROJECT_REPORT](./doc/PROJECT_REPORT.pdf) document
