This is a project that create Generative Adversarial Nets (GANs) from scratch with PyTorch. In this project, I would like to have a first glance on how GANs works and have a better understanding of GANs. Later on, I will test the ability of GANs on 2D image generation. The final goal of the whole GAN project will be test on the ability of GANs to generate 3D model (.stl file).

GANs consist of 2 networks playing an adversarial game against each other

Generator
==============
- Produce fake data and tries to trick the Discriminator


Discriminator
==================
- Inspect the fake data and determines if it's real or fake


1. Randomly initialized
2. Trained simultaneously
    Objective: Minimize 2 losses (2 optimizer)
    BCE Loss

Image dimensions:
Number of channels: 1
Height: 28
Width: 28

Original image size (before transforms):
Height: 28
Width: 28

Pixel value range:
Min pixel value: 0.0
Max pixel value: 1.0

GANs (CNN)
==========
This will be a GANs that consist of CNN.
