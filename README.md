# 3D-Reconstruction through Monocular images 

This repository works upon training a neural network for estimating depth from monocular images that can later be used for 3D reconstruction of the scene.

The companion blog post can be found [here](https://medium.com/@mankaran32/making-a-pseudo-lidar-with-cameras-and-deep-learning-e8f03f939c5f).

![Results](https://i.ibb.co/Ws3CFMf/image.png)

The work is based upon [this paper.](https://arxiv.org/abs/1812.11941)

## Data

13 GB of Data uploaded by [Raghav Prabhakar](https://github.com/Raghav1503)  can be found [here](https://drive.google.com/file/d/1i5Y7Nd-DaWGord9ai-ngT3cn5Pa9az6p/view?usp=sharing)

The dataset contains RGB images and their corresponding depth maps encoded in CARLA's RGB format.

## Training

- Download and place the depth and RGB images in their corresponding folders inside the 'data' directory. 
- Install the dependencies used in the project.
- Run `python train.py` to start training.

## Results on unseen data

![3D-Result](https://i.ibb.co/kKDVkKK/image.png)

Full video can be found [here](https://i.ibb.co/ZVFDym5/ezgif-6-9a8ed53179c8.gif)

## Pretrained Models
The network was collaboratively trained by [Raghav Prabhakar](https://github.com/Raghav1503), [Chirag Goel](https://github.com/chiragoel), [Mandeep](https://github.com/M-I-Dx) and me on google colab for 20 Hrs. 
 
Pretrained model (With DepthNorm Training) can be found [here](https://drive.google.com/file/d/1-LUPM8Nt8WCpYICRIBNvYPJVNDPBnCqd/view?usp=sharing).

## TODO

- [x] Add model training scripts.
- [ ] Add data collection scripts for CARLA.
- [ ] Add easy hyperparameter tuning through a seperate hyperparameter file that can easily be edited.
- [ ] Add 3D reconstruction Code.
- [ ] Add model evaluation code.
- [ ] Train a model without DepthNorm.
- [ ] Add more exception handling .
- [ ] Add CLI for easy argument passing. 
