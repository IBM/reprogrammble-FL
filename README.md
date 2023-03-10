# Reprogrammable-FL: Improving Utility-Privacy Tradeoff in Federated Learning via Model Reprogramming




# Reprogrammable - FL
![image](https://user-images.githubusercontent.com/91891697/209747270-2b54c3c7-9737-4ed5-ae1c-f95cfbabab94.png)


# The Structural Difference of Reprogrammable - FL with baselines
![image](https://user-images.githubusercontent.com/91891697/209747136-2f2bf9c0-600f-4585-8eb1-cb804f22b5c0.png)






# Environment 

pip install opacus

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

pip install medmnist (optional for the medmnist dataset)


# Training the centralized setting

python MR_Centralized.py ( For running the Reprogrammable  Algorithm with DP SGD)

python BL_TS_Cent.py ( For running the Training from Scratch Algorithm with DP SGD)

python BL_FF_Cent.py ( For running the Fully Finetuning Algorithm with DP SGD)

python BL_PS_Cent.py ( For running the Partial Finetuning Algorithm with DP SGD)


# Training the federated setting

python reprog_fl.py ( For running the Reprogrammable FL Algorithm with DP SGD)

python bl_scratch.py ( For running the Training from Scratch Algorithm with DP SGD)

python bl_ff.py ( For running the Fully Finetuning Algorithm with DP SGD)

python bl_pf.py ( For running the Partial Finetuning Algorithm with DP SGD)






# References

Published in IEEE Conference on Security and Trustworthy Machine Learning 2023

https://openreview.net/forum?id=00EiAK1LHs


