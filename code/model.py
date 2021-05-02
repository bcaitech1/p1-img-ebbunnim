import torch
import torch.nn as nn
import torchvision.models as models

################ resnet50 ##################
# classifier re-difinition & initialization
# feature 부분 freeze 안함. 
model_ft = models.resnet50(pretrained=True)
model_ft.fc = nn.Linear(in_features=2048, out_features=18, bias=True) 
# fc
model_ft.fc.weight.data.normal_(0, 0.01)
model_ft.fc.bias.data.zero_()
