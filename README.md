# Deep Generative Model - Project: Medical Image Generation
## Overview
This project, part of the "Deep Generative Model" course (24ss), aims to implement and train Variational Autoencoders (VAEs) for generating medical images from the MedMNIST dataset. The objective is to investigate different VAE variants for tasks such as disentanglement and conditional training across diverse medical imaging modalities.
## Usage
### VAE Training
To view available user-specified parameters, use the -h flag:
```
python training.py -h
```
## VAE-based Medical Image Generator
### Idea:
Generate images from different imaging modalities by training on MedMNIST dataset

### Basic Task:
Implement and train VAE to generate medical images from MedMNIST dataset

### Extension:
Implement VAE variant for 
- (a) disentanglement 
- (b) conditional training 
- (c) evaluate for >= 3 different modalities (e.g. X-ray, Pathology, Dermatology)

## MedMNIST dataset
- website: [MedMNIST](https://medmnist.com/)
- data: [Zenodo](https://zenodo.org/records/10519652)
- github: [MedMNIST GitHub Repository](https://github.com/MedMNIST/MedMNIST)
- paper:  [MedMNIST: A Large-scale Medical Image Benchmark for Medical Image Analysis](https://arxiv.org/abs/2110.14795)
- tutorial: [Getting Started with MedMNIST](https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb)