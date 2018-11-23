## 6.GAN_image_SR

Here we should the implementation of image super resolution with GAN.

We trained a pair of deep learning models: a generator network and a discriminator network. Given the low-resolution images, the generator network outputs the super-resolution images. The discriminator tries to distinguish the model-generated super-resolution images and the actual high-resolution ones. The discriminator network competes with the generator network so that it can push the generator network to produce the super-resolution images as real as possible. During training, we train both the two networks at the same time to make both of them better. Our ultimate goal is to make the generator network output super-resolution images very close to the real high-resolution ones even if the input low-resolution images are previously unseen in the training data. In terms of the dataset, we use the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). We use ResNet as the generator and VGGNet as the discriminator. We also add the perceptual loss to the loss function, which stabilizes the high level representation of the super resolution images.  We use Tensorflow combined with a higher-level package, Tensorlayer, to implement the GAN idea. 

If you want to re-train a new model from scratch, you can refer to [srgan](https://github.com/tensorlayer/srgan).



### Reference
* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802), https://github.com/tensorlayer/srgan
* [DLBI: deep learning guided Bayesian inference for structure reconstruction of super-resolution fluorescence microscopy](https://academic.oup.com/bioinformatics/article/34/13/i284/5045796), https://github.com/lykaust15/DLBI
