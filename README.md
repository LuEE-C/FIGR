# FIGR
Few-shot Image Generation with Reptile

![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/MNIST_50k_red.png)
![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/omniglot_generated_140000_red.png)
![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/icon_80000_red_tower.png)


The gist of this project is that the Reptile meta-learning algorithm is compatible with the GAN setup, unlike the more popular MAML meta-learning algorithm. We train GAN's for few-shot image generation on previously unseen classes on images through this approach.

![alt text](https://github.com/OctThe16th/FIGR/blob/master/images/figr.png)

# FIGR-8
The project also includes a new dataset for few-shot image generation, FIGR-8. A dataset containing 18,409 classes of at least 8 images each for a totla of 1,548,944 images. It can be found here https://github.com/marcdemers/FIGR-8 and here bit.ly/FIGR-8 and is downloaded automatically when running this code like so:

    $ python train.py --dataset FIGR8

# Installation

    $ git clone https://github.com/OctThe16th/FIGR.git
    $ cd FIGR
    $ pip install -r requirements.txt
   
# Usage

    $ python train.py --dataset Mnist & tensorboard --logdir Runs/

For the different command line options, simply write:

    $ python train.py --help
    


If you use this code for your own projects, please consider __citing the following paper__:

	@article{FIGR2019,
	author = {Louis Clouâtre and Marc Demers},
	title = {FIGR: Few-shot Image Generation with Reptile},
	journal = {CoRR},
	volume = {abs/1901.02199},
	year = 2019,
	ee = {http://arxiv.org/abs/1901.02199},
	month = jan,
	archiveprefix = “arXiv”,
	number = “1901.02199v1”,
	eprint = “1901.02199v1”,
	primaryclass = “cs.CV”,
	nonrefereed = “true”
	}
