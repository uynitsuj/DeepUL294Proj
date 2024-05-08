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
import os
import matplotlib.pyplot as plt
from torch_cluster import knn

import PIL.Image
import sys
import pathlib
filepath = str(pathlib.Path(__file__).parent.resolve())
filepath += '/../'
sys.path.append(filepath)
from DUL294P.model_transformer import FoveatedTransformer
from DUL294P.foveation.foveate_image import FoveateImage
from DUL294P.encoders.openclip_encoder import *
import time
from datasets.utils.file_utils import get_datasets_user_agent
import pdb
pdb.set_trace()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_config', type=str,
                   help='the model config file', default='configs/model1.json')
    p.add_argument('--training_config', type=str,
                   help='the training config file', default='configs/train1.json')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--debug', action='store_true', default=False)
    p.add_argument('--num_accumulation_steps', type=int, default=32)
    args = p.parse_args()

    dir = os.path.dirname(os.path.abspath(__file__))

    ### Setup model and train config ###
    print('Checking for model config path: ', os.path.exists(os.path.join(dir, args.model_config)))
    print('Checking for train config path: ', os.path.exists(os.path.join(dir, args.training_config)))
    model_config = json.load(open(os.path.join(dir, args.model_config)))
    training_config = json.load(open(os.path.join(dir, args.training_config)))
    opt_config = training_config['optimizer']
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
        wandb_run = wandb.init(project=wandb_config['project'],
                   config={'model': model_config,
                           'training': training_config,
                           'dataset name': ds_name},
                   save_code=True)
        
    model = FoveatedTransformer(k = model_config["k"], out_channels=5, dropout=model_config["dropout"], num_layers=model_config["num_layers"], channels = model_config["channels"], num_heads=model_config["num_heads"], ratio=model_config["ratio"]).cuda()
    if args.resume:
        model = torch.load(args.resume)
    else:
        if os.path.exists('checkpoints/48-256-3-8.pth'):
            model.load_state_dict(torch.load('checkpoints/48-256-3-8.pth'))
            print('Loaded pretrained model')
        else:
            print('Training from scratch')
            
    clipencoder = OpenCLIPNetworkConfig(device=device).setup()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt_config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training_config['epochs'])
    
    ### Dataloaders ###
    transform = transforms.Compose([
        torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
    ])
    dataset.set_format(type='torch')

    def collate_fn(batch):
        # import pdb; pdb.set_trace()
        return {
            'image': [(x['image']/255.0) for x in batch if (len(x['image'].shape) == 3 and x['image'].shape[0] == 3)],
            # 'clip_image': torch.stack([clipencoder.preprocess(x['image']) for x in batch]),
            'sentences': [x['sentences'] for x in batch]
        }
    trainloader = data.DataLoader(dataset["train"], batch_size, drop_last=True,
                            num_workers=0,
                            shuffle=True,
                            collate_fn=collate_fn)
    
    testloader = data.DataLoader(dataset["validation"], batch_size, drop_last=True,
                            num_workers=0,
                            collate_fn=collate_fn)
    
    cos2 = torch.nn.CosineSimilarity(dim=1) 
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    def train_one_epoch():
        model.train()
        for i, batch in enumerate(tqdm(trainloader)):
            for idx, (image, sentences) in enumerate(zip(batch['image'], batch['sentences'])):
                # import pdb; pdb.set_trace()
                text_embed = clipencoder.model.encode_text(tokenizer(sentences['raw']).cuda())
                start = time.time()
                clipimg = clipencoder.process(image).unsqueeze(0).cuda()
                imgenc = clipencoder.model.encode_image(clipimg).float()
                elapsed = (time.time() - start)
                # print(f"CLIP Elapsed time: {elapsed}(s)")
                # print(f"CLIP Frequency: {1/elapsed}(fps)")
                # import pdb; pdb.set_trace()
                c, h, w = image.size()
                print("Original Image Size", w, h)
                print("Pixel Count", w*h)
                img = image.permute((1,2,0)) # [H, W, C]
                fimg = FoveateImage(w, h)
                foveatedimg, indexes, rs, thetas = fimg.foveate(img)
                foveatedimg = foveatedimg.cuda()
                img = img.cuda()
                rs = rs.cuda()
                thetas = thetas.cuda()
                polar_pos = torch.stack([rs,thetas]).T
                
                start = time.time()
                edge_index = knn(polar_pos, polar_pos, 9) #.flip([0])
                # edge_index = coalesce(torch.cat([edge_index, edge_index.flip([0])]))
                elapsed = (time.time() - start)
                # print(f"KNN Elapsed time: {elapsed}(s)")

                start = time.time()
                fovenc = model(img.unsqueeze(0), foveatedimg, indexes, rs, thetas, edge_index)
                elapsed = (time.time() - start)
                # print(f"Elapsed time: {elapsed}(s)")
                print(f"Frequency: {1/elapsed}(fps)")
                optimizer.zero_grad()
                loss = torch.nn.MSELoss()(fovenc, imgenc)
                loss += torch.nn.MSELoss()(fovenc, text_embed)
                loss = loss / args.num_accumulation_steps
                loss.backward()
                print("Cosine similarity between text and clip image: ", cos2(text_embed, imgenc).data.cpu().numpy())
                print("Cosine similarity between text and foveated image: ", cos2(text_embed, fovenc).data.cpu().numpy())


                # print(idx)
                if ((idx + 1) % args.num_accumulation_steps == 0) or (idx + 1 == len(trainloader)):
                    # print("step")
                    optimizer.step()
                    scheduler.step()
                    wandb_run.log({'train_loss': loss.item()})
            
            torch.save(model.state_dict(), f'checkpoints/model.pth')

    def test_one_epoch():
        model.eval()
        for i, batch in enumerate(tqdm(testloader)):
            for image, sentences in zip(batch['image'], batch['sentences']):
                clipimg = clipencoder.process(image).unsqueeze(0).cuda()
                imgenc = clipencoder.model.encode_image(clipimg).float()
                c, h, w = image.size()
                img = image.permute((1,2,0))
                fimg = FoveateImage(w, h)
                foveatedimg, indexes, rs, thetas = fimg.foveate(img)
                foveatedimg = foveatedimg.cuda()
                img = img.cuda()
                rs = rs.cuda()
                thetas = thetas.cuda()
                polar_pos = torch.stack([rs,thetas]).T
                
                edge_index = knn(polar_pos, polar_pos, 9) #.flip([0])
                fovenc = model(img.unsqueeze(0), foveatedimg, indexes, rs, thetas, edge_index)
                loss = torch.nn.MSELoss()(fovenc, imgenc)
                wandb_run.log({'test_loss': loss.item()})


    epoch = 0
    try:
        while True:
            tqdm.write(f'Epoch {epoch}')
            train_one_epoch()
            test_one_epoch()
            epoch += 1
            tqdm.write('')
            torch.save(model.state_dict(), f'checkpoints/model.pth')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()