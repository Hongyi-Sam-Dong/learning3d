# Assignment 3

## Overview

This project implements Neural Volume Rendering, NeRF, VolSDF, and 3D Gaussian Splatting for CMU 16-825 Learning for 3D Vision.

## Environment

- Python 3.10
- PyTorch
- PyTorch3D
- CUDA

## Running the Code

All parts can be run using the provided `main.py` / launcher scripts.

Example:

```bash
python main_train.py --part 3
python main_train.py --part 6
python main_train.py --part 7.2 --anisotropic

python main_inference.py --part 1
python main_inference.py --part 5
python main_inference.py --part 7.1