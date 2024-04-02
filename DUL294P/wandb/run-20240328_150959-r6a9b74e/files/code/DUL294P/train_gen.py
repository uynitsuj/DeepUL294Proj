#!/usr/bin/env python3


import argparse
import json
# import pickle
import pprint

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from tqdm import trange, tqdm
from datasets import load_dataset
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import matplotlib.pyplot as plt
import PIL.Image
from DUL294P.model_transformer import FoveatedTransformer
from DUL294P.encoders.openclip_encoder import *
import time
from datasets.utils.file_utils import get_datasets_user_agent


USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_config', type=str,
                   help='the model config file', default='configs/model1.json')
    p.add_argument('--training_config', type=str,
                   help='the training config file', default='configs/train1.json')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    args = p.parse_args()

    dir = os.path.dirname(os.path.abspath(__file__))

    ### Setup model and train config ###
    print('Checking for model config path: ', os.path.exists(os.path.join(dir, args.model_config)))
    print('Checking for train config path: ', os.path.exists(os.path.join(dir, args.training_config)))
    model_config = json.load(open(os.path.join(dir, args.model_config)))
    training_config = json.load(open(os.path.join(dir, args.training_config)))
    dataset_config = training_config['dataset']
    opt_config = training_config['optimizer']
    sched_config = opt_config['schedule']
    wandb_config = training_config['wandb']
    print('Model config:')
    pprint.pprint(model_config)
    print('\nTraining config:')
    pprint.pprint(training_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = training_config['batch_size_per_device']

    ds_name = "HuggingFaceM4/COCO"
    dataset = load_dataset(ds_name)

    ### Dataloaders ###
    normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset.set_format(type='torch')

    def collate_fn(batch):
        return {
            'image': [x['image'] for x in batch],
            'sentences': [x['sentences'] for x in batch]
        }
    train_loader = data.DataLoader(dataset["train"], batch_size, drop_last=True,
                             num_workers=dataset_config['num_workers'],
                             persistent_workers=True, collate_fn=collate_fn)
    
    test_loader = data.DataLoader(dataset["validation"], batch_size, drop_last=True,
                             num_workers=dataset_config['num_workers'],
                             persistent_workers=True, collate_fn=collate_fn)

    def train_one_epoch():
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            # import pdb; pdb.set_trace()
            clipimgbatch = []
            for j in range(len(batch['image'])):
                clipimgbatch.append(clipencoder.preprocess(batch['image'][j]))
            clipimgbatch = torch.stack(clipimgbatch).to(device)
            img = clipencoder.preprocess(clipimgbatch).unsqueeze(0).half().to(device)
            import pdb; pdb.set_trace()
            imgenc = clipencoder.model.encode_image(img).float()
            


    if wandb_config['use_wandb'] == 1:
        print('Logging on wandb')
        import wandb
        wandb.init(project=wandb_config['project'],
                   config={'model': model_config,
                           'training': training_config,
                           'dataset name': ds_name},
                   save_code=True)
        
    model = FoveatedTransformer(k = model_config["k"], out_channels=5, dropout=model_config["dropout"], num_layers=model_config["num_layers"], channels = model_config["channels"], num_heads=model_config["num_heads"], ratio=model_config["ratio"])
    if args.resume:
        model = torch.load(args.resume)
    else:
        if os.path.exists('checkpoints/model.pth'):
            model.load_state_dict(torch.load('checkpoints/model.pth'))
            print('Loaded pretrained model')
        else:
            print('Training from scratch')
            
    clipencoder = OpenCLIPNetworkConfig(device=device).setup()


    epoch = 0
    try:
        while True:
            tqdm.write(f'Epoch {epoch}')
            train_one_epoch()
            epoch += 1
            tqdm.write('')
            torch.save(model.state_dict(), f'checkpoints/model.pth')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    # import multiprocessing as mp
    # mp.set_start_method('spawn')
    main()