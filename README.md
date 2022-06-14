# Orchid219_classification
Competition URL: https://tbrain.trendmicro.com.tw/Competitions/Details/20 (public 15th, private 14th)

# Getting started
- Clone this repo to your local
``` bash
git clone https://github.com/come880412/Orchid219_classification
cd Orchid219_classification
```

## Computer Equipment
- System: Ubuntu20.04
- Pytorch version: Pytorch 1.7 or higher
- Python version: Python 3.7
- Testing:  
CPU: AMR Ryzen 7 4800H with Radeon Graphics
RAM: 32GB  
GPU: NVIDIA GeForce RTX 1660Ti 6GB  

- Training (TWCC):  
CPU: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
RAM: 256GB
GPU: NVIDIA GeForce RTX 3090 24GB * 2

## Packages
Please read the "requirement.txt" for the details.

## Download & preprocess dataset
- You should prepare the dataset from [here](https://tbrain.trendmicro.com.tw/Competitions/Details/20), and put the dataset on the folder `../dataset/Orchid219`. After doing so, please use the following command to do data preprocessing.
``` bash
python3 preprocessing.py 
```
- Note: please modify the dataset path on the script `preprocessing.py`.

## Download pretrained models
- Please download the pretrained models from [here](https://drive.google.com/drive/folders/1vjMn-smi6Fj7JLQ--BHu3vbFe9HTOooG?usp=sharing), and put the models on the folder `./public_model`.

## Folder structure
``` 
Orchid
├── Orchid219_classification/ 
├── dataset/ 
  ├── Orchid219
    ├── images
    ├── private_and_public
      ├── public images and private images
    ├── submission_template.csv

``` 

## Inference
- After preparing the dataset and downloading the pretrained models, you could use the following command to generate the .csv file, which is the best public score on our submission.
``` bash
bash inference.sh
```

## Training
- In this competition, we use three models, including swin_large, beit_large_384, convnext_xlarge [1]. You could check the training detail on the script "train.py", and train all models using the following command.
``` bash
bash train.sh
```

# References
[1] https://github.com/rwightman/pytorch-image-models
