# OCR Document Recognition System

GPU-accelerated OCR pipeline for industrial document processing, achieving 100% accuracy and saving 98.33% manual time.

## Features
- PaddleOCR v5 GPU inference for PDF / image / logo recognition
- CV image preprocessing (grayscale, histogram equalization) to improve recognition quality
- Structured output to OracleDB
- SMB network file access for factory shared drives

## Tech Stack
Python, FastAPI, PaddleOCR, PaddlePaddle-GPU, OpenCV, Docker Compose (NVIDIA CUDA)
