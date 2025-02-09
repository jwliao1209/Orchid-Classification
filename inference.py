import os, csv
import argparse
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
from src.config import *
from src.trainer import compute_acc
from src.utils import *
from sklearn.metrics import classification_report
import ttach as tta


def get_test_loader(args, mode):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_transform = Compose([
        CenterCrop(480),
        ToTensor(),
        Normalize(mean, std),
    ])

    data_dir = './datasets'
    test_set = datasets.ImageFolder(os.path.join(data_dir, mode), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=128)
    img_list = [os.path.basename(list(test_set.imgs[i])[0]) for i in range(len(test_set))]

    return test_loader, img_list



def get_topk_ckpt(weight_path, topk):
    topk_ckpt = []

    for c, t in zip(weight_path, topk):
        weight_list = sorted(
            glob.glob(os.path.join('checkpoint', c, 'weight', '*.pth')),
            key=lambda x: float(x[-10:-4]), reverse=True)

        topk_ckpt += weight_list[:t]

    return topk_ckpt


def init_model(models, ckpts, device):
    for m, c in zip(models, ckpts):
        m.load(c)
        m.to(device)
        m.eval()

    return models


def test(args):
    public_loader,  public_list = get_test_loader(args, 'public')
    private_loader, private_list = get_test_loader(args, 'private')
    
    models = [get_model(args) for i in range(sum(args.topk))]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    topk_ckpt = get_topk_ckpt(args.checkpoint, args.topk)
    models = init_model(models, topk_ckpt, device)

    img_list = public_list + private_list

    pred_list = []
    public_bar = tqdm(public_loader, desc=f'Public')
    private_bar = tqdm(private_loader, desc=f'Private')

    with torch.no_grad():
        for batch_data in public_bar:
            image, label = batch_data
            image = image.to(device)

            tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.FiveCrops(args.img_size, args.img_size)
                ])

            pred = []
            for m in models:
                tta_model = tta.ClassificationTTAWrapper(m, tta_transforms)
                pred.append(torch.nn.functional.softmax(tta_model(image), dim=1))

            pred = torch.mean(torch.stack(pred, dim=0), dim=0)
            pred = pred.argmax(dim=1)

            for i in range(pred.size(0)):
                pred_list.append(int(pred[i].cpu().numpy()))

            del image, pred

    with torch.no_grad():
        for batch_data in private_bar:
            image, label = batch_data
            image = image.to(device)

            tta_transforms = tta.Compose([
                tta.HorizontalFlip(),
                tta.FiveCrops(args.img_size, args.img_size)
                ])

            pred = []
            for m in models:
                tta_model = tta.ClassificationTTAWrapper(m, tta_transforms)
                pred.append(torch.nn.functional.softmax(tta_model(image), dim=1))

            pred = torch.mean(torch.stack(pred, dim=0), dim=0)
            pred = pred.argmax(dim=1)

            for i in range(pred.size(0)):
                pred_list.append(int(pred[i].cpu().numpy()))

            del image, pred

    with open('./submit.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'category'])
        for i in range(len(img_list)):
            l = [img_list[i], pred_list[i]]
            writer.writerow(l)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default=['05-12-05-36-53'],
                        nargs='+', help='weight path')
    parser.add_argument('--topk', type=int,
                        default=[3],
                        nargs='+', help='weight of topk accuracy')
    
    args = parser.parse_args()
    config = load_json(os.path.join('checkpoint', args.checkpoint[0], 'config.json'))
    args = argparse.Namespace(**vars(args), **config)
    test(args)
