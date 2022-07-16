import torch
import argparse
import logging
import os
import datetime
import csv
import matplotlib.pyplot as plt
import warnings

from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics import AUROC, ROC
from pathlib import Path
from tqdm import tqdm

from model import Model
from dataset import BmdDataset

ROOT = Path(os.getcwd())


def train(args):
    val_ratio, epochs, batch_size, imgsz, lr = args.val_ratio, args.epochs, args.batch_size, args.imgsz, args.lr
    dataset_dir = args.dataset
    save_checkpoint, cos_lr, roc_curve, amp = args.save_checkpoint, args.cos_lr, args.roc_curve, True

    seoul = datetime.timezone(datetime.timedelta(hours=9)) # Timezone infomation
    date_time = datetime.datetime.now(seoul).strftime('%Y-%m-%d-%H-%M-%S')
    ckp_dir = Path('./runs/train/' + date_time)

    model = Model()
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)

    # Create datasets
    train_transform = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.RandomAutocontrast(p=0.1),
        transforms.RandomAdjustSharpness(2, p=0.1),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = BmdDataset(dataset_dir=dataset_dir, transform=train_transform, aug=True)

    # Split up datasets into train/validation set
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4, nesterov=True)

    # Scheduler
    if cos_lr:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr*0.1)

    # Gradient Scaler for Automated Mixed Precision
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Displaly training setting
    logging.info(f'''Starting training:
    Epochs:              {epochs}
    Batch size:          {batch_size}
    Learning rate:       {lr}
    Training size:       {n_train}
    Validation size:     {n_val}
    Device:              {device}
    Checkpoints:         {save_checkpoint}
    Cosine LR scheduler: {cos_lr}
    Mixed Precision:     {amp}
    ''')

    # Start training
    best_train_acc, best_val_acc, update = 0, 0, False
    log = '0,,'
    
    ckp_dir.mkdir(parents=True, exist_ok=True)
    if roc_curve: (ckp_dir / 'ROC Curve').mkdir(parents=True, exist_ok=True)
    
    f = open(ckp_dir / 'result.csv', 'w')
    f.write('EPOCH,Train LOSS,Train ACC,BEST Train ACC,Valid LOSS,Valid ACC,BEST Valid ACC,AUC\n')
    f.close()
    
    for epoch in range(epochs+1):
        print('\n------------------------')
        print('EPOCH {}/{}'.format(epoch, epochs))
        print('------------------------')

        auc_roc_metric = AUROC(num_classes=3, average=None)
        roc_metric = ROC(num_classes=3)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloaders_dict[phase].dataset.transform = train_transform
                dataloaders_dict[phase].dataset.aug = True
            else:
                model.eval()
                dataloaders_dict[phase].dataset.transform = valid_transform
                dataloaders_dict[phase].dataset.aug = False

            step, epoch_loss, epoch_corrects = 0, .0, 0
            
            # Skip training on epoch=0 to check accuracy when not learning
            if (epoch == 0) and (phase == 'train'):
                continue
            
            pbar = tqdm(enumerate(dataloaders_dict[phase]), total=len(dataloaders_dict[phase]), desc=f'{phase} Epoch {epoch}/{epochs}', unit=' batch')
            for idx, (inputs, labels) in pbar:
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=amp):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            optimizer.zero_grad()
                            grad_scaler.scale(loss).backward()
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                            if cos_lr: scheduler.step()

                        step += 1
                        epoch_loss += loss.item()
                        epoch_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    pbar.set_postfix(**{'LOSS': loss.item()} )

                if phase == 'val':
                    auc_roc_metric(outputs, labels)
                    roc_metric(outputs, labels)
            
            epoch_loss = epoch_loss / step
            epoch_acc = epoch_corrects.item() / (step * batch_size) 


            if phase == 'train':
                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc
                    update = True
                else:
                    update = False
                pbar.set_postfix(**{'LOSS': epoch_loss} )
                print(f'LOSS: {epoch_loss:.4f} ACC: {epoch_acc:.4f} BEST ACC: {best_train_acc:.4f}\n')
                log = ','.join( [str(epoch),str(epoch_loss),str(epoch_acc),str(best_train_acc)] )

            if phase == 'val':
                if update and (epoch_acc == best_val_acc):
                    save_model(model=model, ckp_dir=ckp_dir, acc=epoch_acc, name='best.pth')
                elif best_val_acc < epoch_acc:
                    best_val_acc = epoch_acc
                    logging.info('Best Validation Accuracy: {:.4f}'.format(epoch_acc))
                    save_model(model=model, ckp_dir=ckp_dir, acc=epoch_acc, name='best.pth')

                print(f'{phase} LOSS: {epoch_loss:.4f} ACC: {epoch_acc:.4f} BEST ACC: {best_val_acc:.4f}\n')
                log += ','.join( [',',str(epoch_loss),str(epoch_acc),str(best_val_acc)] )
                
                # Calculate AUROC
                for i, (auroc, fpt, tpr, thresholds) in enumerate(zip(auc_roc_metric.compute(), *roc_metric.compute())):
                    print(f'Class {i} AUC: {auroc:.5f}')
                    size = min(len(fpt), len(tpr))
                    if roc_curve == True:
                        plt.plot(fpt[:size].cpu(), tpr[:size].cpu())
                        plt.savefig(ckp_dir / 'ROC Curve' / ('epoch'+str(epoch)+'_'+str(i)+'.jpg'))
                        plt.cla()
                    log += ','+str(auroc.item())
                print('\n')
                log += '\n'
                save_log( (ckp_dir / 'result.csv'), log)

                if epoch % save_checkpoint == 0:
                    save_model(model=model, ckp_dir=ckp_dir, acc=epoch_acc, epoch=epoch, name='epoch'+str(epoch)+'.pth' )


def save_model(model, ckp_dir, acc=None, epoch=None, name='best.pth'):
    if epoch != None:
        logging.info(f'Saving model Checkpoint {epoch}')
    else:
        logging.info(f'Saving model (ACC {acc:.4f}) to {str(ckp_dir / name)}')
    torch.save({'model_state_dict':model.state_dict(), 'acc':acc}, ckp_dir / name )


def save_log(path, log):
    f = open(path, 'a')
    f.write(log)
    f.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pth', help='pretrained model path')
    parser.add_argument('--dataset', type=str, default=ROOT / 'dataset/train', help='dataset path')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=224, help='image size for training')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--save-checkpoint', type=int, default=1, help='model save cycle')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--roc-curve', action='store_true', help='Save ROC Curve')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_opt()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    warnings.filterwarnings(action='ignore')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)