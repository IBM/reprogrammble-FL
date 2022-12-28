# Reprogrammable-FL: Improving Utility-Privacy Tradeoff in Federated Learning via Model Reprogramming

![image](https://github.com/IBM/reprogrammble-FL/blob/main/dom_final_4.pdf)







# Environment 

pip install opacus

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

pip install medmnist (optional for the medmnist dataset)


# Training

python reprog_fl.py ( For running the Reprogrammable FL Algorithm with DP SGD)

python bl_scratch.py ( For running the Training from Scratch Algorithm with DP SGD)

python bl_ff.py ( For running the Fully Finetuning Algorithm with DP SGD)

python bl_pf.py ( For running the Partial Finetuning Algorithm with DP SGD)



# FAQ

Q1) How are the hyperparameters chosen?

We provide the paper as given in the 




# References

Published in IEEE Conference on Security and Trustworthy Machine Learning 2023

https://openreview.net/forum?id=00EiAK1LHs


