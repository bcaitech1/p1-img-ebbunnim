# Readme

# Description

This repo was generated for participating in the [upstage contest](http://boostcamp.stages.ai/competitions/1/overview/description). I got 79.9524% Accuracy and 0.7528 F1-score in public LB, 79.1429% Accuracy and 0.7333 F1-score in final LB.

- Goal : `Image classification`    
We need a system that automatically identifies whether this person is wearing a mask or not, and whether he or she is wearing it correctly, just by using the image of the person's face shown on the camera. 

- Transfer-learning : Custom model was finetuned-model with pre-trained `resnet50 model`. Experiments were conducted with changing `data augmentation`, `loss`, `optimizer`, `lr scheduler` etc.
```txt
{   
    "seed": 42,
    "epochs": 20,
    "resize": [224,224],
    "batch_size": 64,
    "valid_batch_size": 64,
    "model": "resnet50(pretrained=True)",
    "optimizer": "Adam",
        "scheduler" : "StepLR",
    "gamma" : 0.1,
    "weight_decay": 0.0005,
    "lr": 0.0001,
    "val_ratio": 0.9,
    "criterion": "cross_entropy",
    "lr_decay_step": 7,
}
```

# Installation

`pip install -r requirements.txt` at ./code dir.
```txt
torch==1.6.0
torchvision==0.7.0
tensorboard==2.4.1
pandas==1.1.5
opencv-python==4.5.1.48
scikit-learn~=0.24.1
matplotlib==3.2.1
```

# Contraints
- In this repo, `input directory doesn't exist!`
    - `train.csv` and `submission.csv` was not uploaded. But the column info must contains `ImageID`, `ans` header. 

# Improvements
- ~~Things to improve.~~
- Will be updated.