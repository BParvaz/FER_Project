
# my project!

## what is it?

facial expression recognition software.

-   based on ResNet18
    
-   implements conditional WGAN-GP for data augmentation
    
-   planned diffusion model implementation to compare approaches, use cases, and limitations
    

## why?

-   FER has seen [significant development](https://www.mdpi.com/2695726) recently, while still offering room for growth
    
-   there are papers exploring medical applications, such as [detecting depression](https://jestec.taylors.edu.my/Special%20Issue%20on%20ICIT2022_3/ICIT2022_3_18.pd)
    
-   it’s a personal interest of mine that i believe has space for diverse use cases
    

## how's it going?

-   ~~good, how about you?~~
    
-   i’m really happy with how the project has progressed so far
    
-   as a learning experience, it’s allowed me to branch into different areas of software development (notably HPC and integrating independent research into my work)

## key components

 - [FER model](https://github.com/BParvaz/FER_Project/blob/main/model.ipynb)
 - [WGAN-GP](https://github.com/BParvaz/FER_Project/blob/main/CGAN/train.py)
 - [dataset](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
 - [email](mailto:bahramaparvaz@outlook.com)

## todo (from necessary to aspirational)

1.  rigorous testing of existing code
    
2.  implement diffusion model
    
3.  improve existing model performance
    
4.  improve modularity to make dataset/use-case changes easier
    
5.  streamline usage from command-line scripts to a cleaner interface
    
6.  eventually add representative data samples to the repository

thanks to ainur for supervision and guidance, and to the team managing UOM’s HPC resources for supporting the computational work behind this project.
