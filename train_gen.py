#!/usr/bin/env python3


import argparse
import json
import pickle
import pprint

import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.transforms.functional import pil_to_tensor
from tqdm import trange, tqdm
import webdataset as wds
from datasets import load_dataset
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import matplotlib.pyplot as plt
import PIL.Image

from datasets import load_dataset
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
    p.add_argument('model_config', type=str,
                   help='the model config file')
    p.add_argument('training_config', type=str,
                   help='the training config file')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    args = p.parse_args()

    config = json.load(open(args.model_config))
    training_config = json.load(open(args.training_config))
    dataset_config = training_config['dataset']
    opt_config = training_config['optimizer']
    sched_config = opt_config['schedule']
    wandb_config = training_config['wandb']
    print('Model config:')
    pprint.pprint(config)
    print('\nTraining config:')
    pprint.pprint(training_config)
    print()
    batch_size = training_config['batch_size_per_device']

    num_threads = 20
    dataset = load_dataset("conceptual_captions")

    train_loader = data.DataLoader(dataset["train"], batch_size, drop_last=True,
                             num_workers=dataset_config['num_workers'],
                             persistent_workers=True)
    
    test_loader = data.DataLoader(dataset["validation"], batch_size, drop_last=True,
                             num_workers=dataset_config['num_workers'],
                             persistent_workers=True)

    def train_one_epoch():
        for i, batch in enumerate(tqdm(train_loader)):
            
            for id, img in enumerate(batch['image_url']):
                # import pdb; pdb.set_trace()
                if "image_filtered" not in batch:
                    batch["image_filtered"] = []
                    batch["text_filtered"] = []
                pil_img = fetch_single_image(img)
                if pil_img is None:
                    continue
                batch["image_filtered"].append(pil_to_tensor(pil_img))
                batch["text_filtered"].append(batch["caption"][id])
                print(batch["image_filtered"][-1])
                print(batch["text_filtered"][-1])
                import pdb; pdb.set_trace()

            import pdb; pdb.set_trace()


    if wandb_config['use_wandb'] == 0:
        import wandb
        wandb.init(project=wandb_config['project'],
                   config={'model': config,
                           'training': training_config},
                   save_code=True)

    epoch = 0
    try:
        while True:
            tqdm.write(f'Epoch {epoch}')
            train_one_epoch()
            epoch += 1
            tqdm.write('')
            # save()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()