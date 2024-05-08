#!/usr/bin/env python3


import argparse
import json

# import pickle
import pprint

import torch
from torch.utils import data
from torchvision.transforms import v2 as T
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm
from datasets import load_dataset
import os
import logging
from open_clip.loss import DistillClipLoss
import sys

torch.multiprocessing.set_sharing_strategy("file_system")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


# from DUL294P.model_transformer import FoveatedTransformer
# from DUL294P.foveation.foveate_image import FoveateImage
from DUL294P.encoders.openclip_encoder import *
from DUL294P.mae.models_mae import MaskedAutoencoderViT, partial
import time

# from datasets.utils.file_utils import get_datasets_user_agent

def mae_inference(model, img, training_config):
    latent, mask, ids_restore = model.forward_encoder(
        img, training_config["masking_prob"]
    )
    pred_mae, pooled_mae = model.forward_decoder(latent, ids_restore)
    return pooled_mae, pred_mae

def clip_inference(model, img, training_config):
    pooled_mae, pred_mae = model(img) 
    return pooled_mae, pred_mae

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model_config",
        type=str,
        help="the model config file",
        default="configs/mae_model.json",
    )
    p.add_argument(
        "--training_config",
        type=str,
        help="the training config file",
        default="configs/train1.json",
    )
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--resume", type=str, help="the checkpoint to resume from")
    p.add_argument("--debug", action="store_true", default=False)
    args = p.parse_args()
    dir = os.path.dirname(os.path.abspath(__file__))

    logging.basicConfig(
        filename=f"{dir}/logs/train.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    ### Setup model and train config ###
    print(
        "Checking for model config path: ",
        os.path.exists(os.path.join(dir, args.model_config)),
    )
    print(
        "Checking for train config path: ",
        os.path.exists(os.path.join(dir, args.training_config)),
    )
    model_config = json.load(open(os.path.join(dir, args.model_config)))
    training_config = json.load(open(os.path.join(dir, args.training_config)))
    opt_config = training_config["optimizer"]
    wandb_config = training_config["wandb"]
    print("Model config:")
    pprint.pprint(model_config)
    print("\nTraining config:")
    pprint.pprint(training_config)
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    batch_size = training_config["batch_size_per_device"]

    ds_name = "HuggingFaceM4/COCO"
    dataset = load_dataset(ds_name)
    # if wandb_config["use_wandb"] == 1:
    #     print("Logging on wandb")
    #     import wandb

    #     wandb.init(
    #         project=wandb_config["project"],
    #         config={
    #             "model": model_config,
    #             "training": training_config,
    #             "dataset name": ds_name,
    #         },
    #         save_code=True,
    #     )

    if model_config["model"] == "MAE":
        model = MaskedAutoencoderViT(
            patch_size=model_config["patch_size"],
            embed_dim=model_config["embed_dim"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            decoder_embed_dim=model_config["decoder_embed_dim"],
            decoder_depth=model_config["decoder_depth"],
            decoder_num_heads=model_config["decoder_num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            out_chans=model_config["out_chans"],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            return_pooled=True,
        )

        model.load_state_dict(
            torch.load(model_config["pretrained_path"])["model"], strict=False
        )
        inf_func = mae_inference
    elif model_config["model"] == "CLIP":
        model = OpenCLIPNetworkConfig(device=device).setup().float().model.visual
        model.patch_dropout = PatchDropout(model_config["patch_dropout"])
        inf_func = clip_inference
            
    else:
        raise ValueError("Model not supported")
        # model = FoveatedTransformer(
        #     k=model_config["k"],
        #     out_channels=5,
        #     dropout=model_config["dropout"],
        #     num_layers=model_config["num_layers"],
        #     channels=model_config["channels"],
        #     num_heads=model_config["num_heads"],
        #     ratio=model_config["ratio"],
        # )

    clipencoder = OpenCLIPNetworkConfig(device=device).setup().eval().to(device).float()
    model.to(device)
    model.float()

    ### Dataloaders ###
    dataset.set_format(type="torch")

    def collate_fn(batch):

        images = []
        for ex in batch:
            img = ex["image"]
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            images.append(img / 255.0)

        return {
            "image": images,
            "clip_image": torch.stack([clipencoder.process(x) for x in images]),
            "sentences": [x["sentences"] for x in batch],
        }

    trainloader = data.DataLoader(
        dataset["train"],
        batch_size,
        drop_last=False,
        num_workers=6,
        collate_fn=collate_fn,
    )

    valloader = data.DataLoader(
        dataset["validation"],
        batch_size,
        drop_last=False,
        num_workers=6,
        collate_fn=collate_fn,
    )
    testloader = data.DataLoader(
        dataset["test"],
        batch_size,
        drop_last=False,
        num_workers=6,
        collate_fn=collate_fn,
    )
    
    # @profile

    @torch.no_grad()
    def val_one_epoch(loader):
        model.eval()
        losses = []
        times = []
        clip_sims = []
        for batch in loader:
            pass
        pbar = tqdm(enumerate(loader), total=len(loader))
        import pdb
        pdb.set_trace()
        for i, (batch) in pbar:
            import pdb
            pdb.set_trace()
            img = batch["clip_image"].to(device)
            pooled_clip, pred_clip = clipencoder.model.encode_image(img)
            clip_sent = clipencoder.model.encode_text(
                clipencoder.tokenizer([s["raw"] for s in batch["sentences"]]).to(
                    device
                ),
                normalize=True,
            )
            
            pooled_mae, _ = inf_func(model, img, training_config)


            # loss = torch.nn.MSELoss()(pred_mae, pred_clip)
            loss = (
                torch.nn.MSELoss()(pooled_mae, pooled_clip)
                * training_config["pooled_loss_weight"]
            )
            loss = loss.item()
            clip_sim = (F.cosine_similarity(pooled_clip, clip_sent) - F.cosine_similarity(pooled_mae, clip_sent)).abs().mean().item()
            losses.append(loss)
            clip_sims.append(clip_sim)
            pbar.set_description(
                f"Loss: {loss:4f}, Text Sim: {clip_sim:4f}, Avg Freq: {(len(times) / (sum(times)+1)):4f}"
            )
            if i < 2:
                for image, sentences in zip(batch["image"], batch["sentences"]):
                    clipimg = clipencoder.process(image).unsqueeze(0).float().to(device)
                    pooled_clip, pred_clip = clipencoder.model.encode_image(clipimg)
                    clip_sent = clipencoder.model.encode_text(
                        clipencoder.tokenizer(sentences["raw"]).to(device),
                        normalize=True,
                    )
                    start = time.time()
                    pooled_mae, _ = inf_func(model, clipimg, training_config)
                    elapsed = time.time() - start
                    # loss = torch.nn.MSELoss()(pred_mae, pred_clip)
                    loss = (
                        torch.nn.MSELoss()(pooled_mae, pooled_clip)
                        * training_config["pooled_loss_weight"]
                    )
                    clip_sim = F.cosine_similarity(pooled_mae, clip_sent).item()
                    # print(clip_sim.shape, clipimg.shape, clip_sent.shape)
                    # losses.append(loss.item())
                    times.append(elapsed)
                    # clip_sims.append(clip_sim)
                    pbar.set_description(
                        f"Loss: {loss.item():4f}, Time: {elapsed:4f}(s), Frequency: {(1/elapsed):4f}(fps), Text Sim: {clip_sim:4f}"
                    )
            else:
                img = batch["clip_image"].to(device)
                pooled_clip, pred_clip = clipencoder.model.encode_image(img)
                clip_sent = clipencoder.model.encode_text(
                    clipencoder.tokenizer([s["raw"] for s in batch["sentences"]]).to(
                        device
                    ),
                    normalize=True,
                )
                
                pooled_mae, _ = inf_func(model, img, training_config)


                # loss = torch.nn.MSELoss()(pred_mae, pred_clip)
                loss = (
                    torch.nn.MSELoss()(pooled_mae, pooled_clip)
                    * training_config["pooled_loss_weight"]
                )
                loss = loss.item()
                clip_sim = (F.cosine_similarity(pooled_clip, clip_sent) - F.cosine_similarity(pooled_mae, clip_sent)).abs().mean().item()
                losses.append(loss)
                clip_sims.append(clip_sim)
                pbar.set_description(
                    f"Loss: {loss:4f}, Text Sim: {clip_sim:4f}, Avg Freq: {(len(times) / (sum(times)+1)):4f}"
                )

        print(f"Mean Loss: {sum(losses) / len(losses)}")
        print(f"Mean Time: {sum(times) / len(times)}")
        print(f"Mean Clip Similarity: {sum(clip_sims) / len(clip_sims)}")
        logging.info(
            f"Val Stats: Loss: {sum(losses) / len(losses)}, Time: {sum(times) / len(times)}, Clip Sim: {sum(clip_sims) / len(clip_sims)}"
        )
        return (
            sum(losses) / len(losses),
            sum(times) / len(times),
            sum(clip_sims) / len(clip_sims),
        )

    epoch = 0
    
    val_one_epoch(testloader)
    import pdb
    pdb.set_trace()
    for _ in range(training_config["epochs"]):
        tqdm.write(f"Epoch {epoch}")
        train_one_epoch()
        loss, val_time, clip_sims = val_one_epoch(valloader)
        epoch += 1
        # tqdm.write("")
        torch.save(model.state_dict(), f"checkpoints/model.pth")


if __name__ == "__main__":
    main()
