# RobustTransferLWF
Adversarially Robust Transfer Learning with LWF loss applied to the deep feature representation (penultimate) layer

Download the dataset you want to transfer to (target) and place it in its corresponding folder. For CIFAR-10 (default), you can run the download_cifar10.sh script.

Download a an adversarially trained CIFAR model using fetch_model.py (for CIFAR100: fetch_model.py adv_trained_cifar100)

run train_with_LWF.py
