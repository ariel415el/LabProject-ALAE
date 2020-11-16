# Comparing AutoEncoders
This repository compares some linear and non-linear (MLP) autoencoders and 
Shows an analysis of their differences

## Auto-Encoders
#### PCA
The first step in learning autoencoders should probably be PCA. It can be proved that PCA is the optimal
linear autoencoder in minimizing reonstruction loss.
Understanding PCA: PCA could be looked at as projected variance maximization 
    or reconstrcion loss minimization problems under the constraint that the encoding Linear
    matrix is orthonormal.
    the PCA foler shows how gradient decent optimization of the above problems leads to the 
    same solution as the analytic PCA solution for the problem

#### ALAE
ALAE is an autoencoder that is trained in adverserial way along with an latent reconstuction error minimization
paper:
I implemented it in a thorough way here: https://github.com/ariel415el/ALAE
I'll use it here comparing it to other auto-encoders

## Comparison
1. Compare the auto-encoders on various datasets by their reconstruction loss on the test data.
2. train classifiers in the encoding learned from classification dataset (e.g Mnist) and compare 
    test accuracy.
   
The main results of step 1:

![alt text](assets/Mnist-trainig.png)

##### Relevant materials #####
- PCA: two sided problem: http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/



# Relevant papers:
- Variational AutoEncoders: https://arxiv.org/abs/1312.6114
- Adverserial AutoEncoders: https://arxiv.org/abs/1511.05644
- AutoEncoders with learned metric: https://arxiv.org/abs/1512.09300
- Adverserial Feature Learning (BiGAN): https://arxiv.org/abs/1605.09782
- PGGANs: https://arxiv.org/pdf/1710.10196.pdf
- StyleGAN: https://arxiv.org/abs/1812.04948
