import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from tqdm import tqdm, trange
import pickle
import PIL.Image as Image
import json
import random
import sys
import clip
import PIL
import random


device = torch.device('cuda:0')
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = clip.tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_Tokenizer = _Tokenizer()


text = "a person on a motorcycle rider jumps in the air"
texts_token = tokenizer(text).to(device)
text_feature = clip_model.encode_text(texts_token)
text_feature /= text_feature.norm(dim=-1,keepdim=True)

path_pic = 'images/000000190756.jpg'  #Pictures from MSCOCO val.
image = Image.open(path_pic)
image = preprocess(image).unsqueeze(0).to(device)
image_features = clip_model.encode_image(image).float()
image_features /= image_features.norm(dim=-1,keepdim=True)

random_feature = torch.rand(1,512).to(device)
# compute cosine similarity in dim=1 
cos_2 = torch.nn.CosineSimilarity(dim=1) 
output_2 = cos_2(text_feature, image_features) 
output_3 = cos_2(image_features, random_feature)
output_4 = cos_2(random_feature,text_feature)
# display the output tensor 
print("\n\nComputed Cosine Similarity for Image and Text in dim=1: ", 
      output_2)
print("\n\nComputed Cosine Similarity for Image and Random in dim=1: ", 
      output_3)
print("\n\nComputed Cosine Similarity for Random and Text in dim=1: ", 
      output_4)