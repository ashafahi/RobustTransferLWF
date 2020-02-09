# Adversarially Robust Transfer Learning
This repo contains parts of the learning without forgetting experiment of the [Adversarially Robust Transfer Learning (ICLR 2020)](https://arxiv.org/abs/1905.08232 "Adversarially Robust Transfer Learning") paper. In the paper, we study and identify the source of robustness and use our finding to do robust transfer learning by distilling the robustness of a teacher network on to the student network. Using adversarially robust transfer learning, we can learn a robust model for a target dataset where we have limited number of training data.
## Adversarially robust transfer learning with the LwF loss
Adversarially Robust Transfer Learning with a learning without forgetting loss applied to the deep feature representation (penultimate) layer. The idea is to regularize the network to maintain its robustness (using the LwF penalty) but allow it also to adopt itself to the target dataset. 
Our scripts are inspired by the Madry Lab [CIFAR10 Adversarial Example Challenge](https://github.com/MadryLab/cifar10_challenge "Madry's CIFAR10 Challenge") and we give them credit for that. 

## running the model from a robust CIFAR-100 to a CIFAR-10 model.
To run the training script, we need to have the target dataset and the source pre-trained model.

Download the dataset you want to transfer to (target) and place it in its corresponding folder. For CIFAR-10 (default), you can run the following to download CIFAR-10 and place it in the correct directory:
```bash
source download_cifar10.sh 
```

Download a pre-trained source model (default is an adversarially trained CIFAR100 model). The robust CIFAR-100 model can be downloaded using fetch_model.py:
```bash
python fetch_model.py adv_trained_cifar100 
```

Given the robust model and the target dataset, you can run the main training script:
```bash
python train_with_LWF.py 
```

## Citing the paper
If you liked the paper, please consider citing it.
```bash
@article{shafahi2019adversarially,
  title={Adversarially robust transfer learning},
  author={Shafahi, Ali and Saadatpanah, Parsa and Zhu, Chen and Ghiasi, Amin and Studer, Christoph and Jacobs, David and Goldstein, Tom},
  journal={arXiv preprint arXiv:1905.08232},
  year={2019}
}
```
