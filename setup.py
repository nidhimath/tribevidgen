from setuptools import setup, find_packages

setup(
    name="neuroadgen",
    version="0.1.0",
    description="Brain-optimised ad video generation using TribeV2 neural reward model",
    author="NeuroAdGen",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.3.0",
        "torchvision",
        "diffusers>=0.30.0",
        "transformers>=4.44.0",
        "accelerate>=0.34.0",
        "peft>=0.12.0",
        "numpy==2.2.6",
        "pyyaml",
        "tqdm",
        "opencv-python",
        "gradio>=4.0.0",
        "wandb",
        "huggingface_hub",
        "moviepy",
    ],
    extras_require={
        "brain": ["tribev2", "nilearn", "pyvista", "vtk"],
        "train": ["deepspeed>=0.15.0"],
        "api": ["fal-client"],
        "dev": ["pytest", "black", "isort"],
    },
    entry_points={
        "console_scripts": [
            "neuroadgen=neuroadgen.inference.generate:main",
        ],
    },
)
