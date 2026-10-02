# Assignment 2

`main_train.py` is the unified training entry point for all Assignment 2 models. It runs all training targets sequentially, automatically resumes from existing checkpoints, and uses CUDA by default.

Run all training:

    python main_train.py

Run a single training target:

    python main_train.py --target q3_1

`main_inference.py` is the unified evaluation and visualization entry point. It runs all inference and evaluation tasks sequentially and generates the corresponding figures and metrics.

Run all inference:

    python main_inference.py

Run a single inference target:

    python main_inference.py --target q3_1