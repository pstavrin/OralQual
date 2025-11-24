# **CSE Oral Qualifying Exam**
This repository contains code for the computational artifact to be presented as part of my PhD oral qualifying exam. Code is written in the `Julia` programming language.

# **Contents**
The artifact implements *ensemble Kalman randomized maximum likelihood estimation*, originally introduced [here](https://arxiv.org/abs/2507.03207), which is a derivative-free method that can be used to solve Bayesian inverse problems.
The artifact also show how to use the method to:
* solve least-squares problems and study convergence of the algorithm
* solve Bayesian inverse problems in the linear Gaussian setting and illustrate convergence to the *exact* Gaussian posterior
* solve nonlinear Bayesian inverse problems and produce approximate samples from the Bayesian posterior
