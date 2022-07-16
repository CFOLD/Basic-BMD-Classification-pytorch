import torch
import argparse
import logging
import os
import datetime
import csv

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from tqdm import tqdm

from model import Model

ROOT = Path(os.getcwd())


def predict(args):
    imgsz, batch_size = args.imgsz, args.batch_size
    dataset_dir = args.dataset

    seoul = datetime.timezone(datetime.timedelta(hours=9)) # Timezone infomation
    date_time = datetime.datetime.now(seoul).strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = Path('./runs/test/' + date_time)

    model = Model()
    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model_state_dict'])
    model.to(device)

    # Create datasets
    test_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(dataset_dir, transform=test_transform)

    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Start testing
    save_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    result = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Testing', unit=' batch')
    for idx, (inputs, labels) in pbar:
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # 라벨 예측
            for pred in preds.flatten():
                result.append(pred.item())

    with open(save_dir / 'result.csv', 'w') as f:
        wr = csv.writer(f)
        wr.writerow(['index', 'label_answer'])
        for idx, prediction in enumerate(result):
            wr.writerow([idx+1, prediction])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pth', help='pretrained model path')
    parser.add_argument('--dataset', type=str, default=ROOT / 'dataset/test', help='dataset path')
    parser.add_argument('--imgsz', type=int, default=224, help='image size for training')
    parser.add_argument('--batch-size', type=int, default=32)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predict(args)