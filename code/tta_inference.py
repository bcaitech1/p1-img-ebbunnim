import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import TestDataset, MaskBaseDataset, AddGaussianNoise, TestDataset_V2
from PIL import Image
from tqdm import tqdm

def load_model(saved_model, num_classes, device):
    model = getattr(import_module("model"), args.model)  # custom : resnet50
    model_path = os.path.join(saved_model, 'best.pth')
    print('model_path: ',model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

@torch.no_grad()
def tta_inference_all(transforms):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(args.model_dir, num_classes, device).to(device)
    model.eval()
    

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    outputs = None
    for i, transform in tqdm(enumerate(transforms)):
        print(f"Transform {i+1}/{len(transforms)}")
        dataset=TestDataset_V2(img_paths,mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=4,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        if i == 0:
            outputs = torch.zeros(len(dataset), 18)
        tmp_outputs = torch.zeros(len(dataset), 18)
        
        # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
        with torch.no_grad():
            for j, img in enumerate(loader):
                img = img.to(device)
                tmp_outputs[j * args.batch_size: (j+1) * args.batch_size] = model(img)
            outputs = outputs + tmp_outputs
    
    outputs /= len(transforms)
    preds = []
    for output in outputs:
        pred_class = torch.argmax(output)
        preds.append(pred_class.cpu().numpy())
    
    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(224,224), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='model_ft', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp38'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    test_transform1 = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std),
        ])

    test_transform2 = transforms.Compose([
            transforms.Resize((224,224),Image.BILINEAR),
            transforms.CenterCrop((224,224)),
            # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            AddGaussianNoise(0.01,0.008)
        ])
    test_transform3 = transforms.Compose([
        transforms.Resize((224,224),Image.BILINEAR),
        transforms.CenterCrop((224,224)),
        # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # AddGaussianNoise(0.01,0.008)
    ])

    test_transforms = [test_transform1, test_transform2, test_transform3]
    tta_inference_all(test_transforms)