# Explorations in Linear Models, NTKs, and Scaling

![image](https://github.com/user-attachments/assets/553d5c34-9981-4cb3-8113-d8a664e28a13)


This repo is dedicated to recaputilating various results relevant to machine learning in idealized settings. I study linear models as well as neural networks trained on classic computer vision datasets, e.g. CIFAR-10 and MNIST.
I also follow "A Solveable Model of Neural Scaling Laws" by Maloney, Roberts, and Sully, which analyzes scaling in the case of random feature models. 

I also look at aspects of linearization such as the Neural Tangent Kernel (NTK). In particular I investigate the factorization of the NTK when a model has multi-dimensional outputs e.g. classification tasks like in MNIST. The figure below demonstrates empirically that the NTK does indeed factorize.

<img width="511" height="413" alt="ntk_fac1" src="https://github.com/user-attachments/assets/0fe50d15-6d60-4fe0-91d8-7f0361878af2" /> 

Furthermore, the NTK spectrum exhibits log-log scaling

<img width="578" height="459" alt="ntkloglog" src="https://github.com/user-attachments/assets/ca2fc82b-e2ef-4e59-b739-6b76c3759cf6" />
