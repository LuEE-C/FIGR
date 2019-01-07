# FIGR
Few-shot Image Generation with Reptile

![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/MNIST_50k_red.png)
![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/omniglot_generated_140000_red.png)
![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/icon_80000_red_tower.png)


The gist of this project is that the Reptile meta-learning algorithm is compatible with the GAN setup, unlike the more popular MAML meta-learning algorithm. We train GAN's for few-shot image generation on previously unseen classes on images through this approach.

![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/figr.png)

# Installation

    $ git clone https://github.com/OctThe16th/FIGR.git
    $ cd FIGR
    $ pip install -r requirements.txt
   
# Usage

    $ python train.py --dataset Mnist & tensorboard --logdir Runs/

For the different command line options, simply write:

    $ python train.py --help
