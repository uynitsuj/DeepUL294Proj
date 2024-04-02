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
from DUL294P.foveation.foveate_image import FoveateImage
from DUL294P.encoders.openclip_encoder import *
import time
from datasets.utils.file_utils import get_datasets_user_agent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_config', type=str,
                   help='the model config file', default='configs/model1.json')
    p.add_argument('--training_config', type=str,
                   help='the training config file', default='configs/train1.json')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--debug', action='store_true', default=False)
    args = p.parse_args()

    dir = os.path.dirname(os.path.abspath(__file__))

    ### Setup model and train config ###
    print('Checking for model config path: ', os.path.exists(os.path.join(dir, args.model_config)))
    print('Checking for train config path: ', os.path.exists(os.path.join(dir, args.training_config)))
    model_config = json.load(open(os.path.join(dir, args.model_config)))
    training_config = json.load(open(os.path.join(dir, args.training_config)))
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

    ### Dataloaders ###
    transform = transforms.Compose([
        torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
    ])
    dataset.set_format(type='torch')

    def collate_fn(batch):
        return {
            'image': [transform(x['image'].permute(2,0,1)/255.0) for x in batch],
            # 'clip_image': torch.stack([clipencoder.preprocess(x['image']) for x in batch]),
            'sentences': [x['sentences'] for x in batch]
        }
    trainloader = data.DataLoader(dataset["train"], batch_size, drop_last=True,
                            num_workers=0,
                            collate_fn=collate_fn)
    
    testloader = data.DataLoader(dataset["validation"], batch_size, drop_last=True,
                            num_workers=0,
                            collate_fn=collate_fn)
    
    def train_one_epoch():
        model.train()
        for i, batch in enumerate(tqdm(trainloader)):
            for j, image, sentences in enumerate(zip(batch['image'], batch['sentences'])):
                clipimg = clipencoder.process(image).unsqueeze(0).half().cuda()
                imgenc = clipencoder.model.encode_image(clipimg).float()
                
                c, h, w = image.size()
                print("Original Image Size", w, h)
                print("Pixel Count", w*h)
                img = image.reshape((h,w,c))
                fimg = FoveateImage(w, h)
                start = time.time()
                foveatedimg, idxs = fimg.foveate(img)
                elapsed = (time.time() - start)
                print(f"Elapsed time: {elapsed}(s)")
                print(f"Frequency: {1/elapsed}(fps)")
                print("Foveated Pixel Count", foveatedimg.shape)

                recon = torch.zeros((h,w,3), dtype=foveatedimg.dtype)
                recon.view(-1, 3)[idxs] = foveatedimg[range(len(foveatedimg))]

                print(f"Compression: {len(idxs)/(w*h)*100}%")
                import pdb; pdb.set_trace()
                
                plt.imshow(recon)
                plt.show()
                import pdb; pdb.set_trace()


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
    main()