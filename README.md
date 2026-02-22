
## Facial Expression Recognition with Generative Augmentation

### Overview

This project implements a facial expression recognition (FER) pipeline based on ResNet-18, with conditional WGAN-GP used for minority-class data augmentation.

The goal is to investigate whether generative models can mitigate class imbalance in FER datasets such as FER2013 and RAF-DB.

A diffusion-based generative model is planned for comparison against the GAN-based approach.

----------

### Motivation

Facial expression recognition has seen significant recent development, while still presenting open challenges — particularly in handling imbalanced emotion classes.

Emerging research explores broader applications of affect recognition, including mental health–related contexts.

This project serves both as a research investigation into generative augmentation and as a systems-level learning experience involving HPC workflows and experimental evaluation.

----------

### Current Status

-   ResNet-18 classifier implemented in PyTorch
    
-   Conditional WGAN-GP integrated for class-specific augmentation
    
-   Multi-run training experiments conducted on institutional HPC (A100/L40S GPUs)
    
-   Ongoing evaluation using macro F1, balanced accuracy, and per-class recall
    

----------

### Roadmap

1.  Rigorous validation and testing of existing training pipeline
    
2.  Implementation of diffusion-based generative baseline
    
3.  Further improvement of generator quality and stability
    
4.  Refactoring for improved modularity across datasets and use cases
    
5.  Improved command-line interface and usability
    
6.  Addition of representative synthetic samples
    

----------

### Acknowledgements

Thanks to Ainur for supervision and guidance, and to the team managing UoM’s HPC resources for supporting the computational work behind this project.