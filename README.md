# GANs for MNIST Digit Generation

This project implements a Generative Adversarial Network (GAN) in PyTorch to generate realistic-looking MNIST digits. GANs consist of two competing neural networks, a **Generator** and a **Discriminator**, which learn together to produce and distinguish between real and generated images.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)

## Project Overview

This project demonstrates a simple GAN architecture to generate images resembling handwritten digits from the MNIST dataset. The generator learns to create realistic images, while the discriminator learns to classify images as real or fake. 

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yusufshihata/GAN.git
    cd project-name
    ```

2. **Set up the environment**:
   - Create a virtual environment (optional but recommended):
     ```bash
     python -m venv env
     source env/bin/activate  # For Linux/macOS
     env\Scripts\activate     # For Windows
     ```
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

1. **Training the GAN**:
    Run the main training script:
    ```bash
    python src/main.py
    ```

    This will train the GAN model on the MNIST dataset, saving generated images at the end of each epoch in the `generated_images` folder.

2. **Viewing Generated Images**:
   - Each epoch, the model generates sample images saved in `generated_images/`.
   - To visualize images, you can use the `show_image` function from `utils/show_imgs.py`.

## Model Architecture

### Generator
The Generator network takes a noise vector as input and applies a series of transpose convolutions to generate an image resembling an MNIST digit.

- **Input**: Random noise vector of size `NOISE_DIM`
- **Output**: 28x28 grayscale image

### Discriminator
The Discriminator network takes an image and attempts to classify it as real or fake.

- **Input**: 28x28 grayscale image
- **Output**: Probability (real/fake)

## Results

The generator improves over epochs, producing increasingly realistic images of handwritten digits. Here’s an example of generated digits after a few training epochs:

![Example](generated_images/epoch_50.png)

## References

1. **Ian Goodfellow et al.** - [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)

---
