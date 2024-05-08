from dataclasses import dataclass, field
from typing import Tuple, Type

import torch
from scipy.stats import norm
import torchvision

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

from DUL294P.encoders.image_encoder import (BaseImageEncoder,
                                         BaseImageEncoderConfig)

@dataclass
class OpenCLIPNetworkConfig(BaseImageEncoderConfig):
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 768
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    device: str = 'cuda'
    output_tokens: bool = True
    masking_prob: float = 0.75

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.clip_model_type, self.clip_model_pretrained)


class PermuteChannels:
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be permuted.
        Returns:
            Tensor: Permuted tensor.
        """
        return tensor.permute(*self.permutation)
    
class OpenCLIPNetwork(BaseImageEncoder):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                PermuteChannels((2,0,1)),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
                
            ]
        )
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        model.visual.output_tokens = self.config.output_tokens
        model.visual.patch_dropout = PatchDropout(self.config.masking_prob)
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to(self.config.device)
        self.clip_n_dims = self.config.clip_n_dims

        # self.positives = self.positive_input.value.split(";")
        self.negatives = self.config.negatives
        with torch.no_grad():
            # tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.config.device)
            # self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to(self.config.device)
            self.neg_embeds = model.encode_text(tok_phrases)
        # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        # assert (
        #     self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        # ), "Positive and negative embeddings must have the same dimensionality"
        # assert (
        #     self.pos_embeds.shape[1] == self.clip_n_dims
        # ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    # def gui_cb(self,element):
    #     self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(self.config.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)

class PatchDropout(torch.nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x
    
class PatchDropoutFov(torch.nn.Module):
    """
    https://arxiv.org/abs/2212.00794

    Modified to include patch selection with a normal distribution bias
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x, center_coord, std_dev, return_indices=False):
        if self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch_size, num_tokens, _ = x.size()

        # Calculate distances from center coordinate
        # get grid of token coordinates
        token_locs = torch.arange(num_tokens)
        distances = ((token_locs // num_tokens**.5) - center_coord[0])**2 \
                               + ((token_locs % num_tokens**.5) - center_coord[1])**2

        # Calculate probabilities based on normal distribution
        # probabilities = torch.exp(-distances / (2 * std_dev**2))
        probabilities = 1/(std_dev*(1+distances/std_dev**2))

        # Normalize probabilities to sum to 1
        probabilities /= probabilities.sum()

        # Determine how many patches to keep based on overall dropout probability
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        # Sample patches to keep based on probabilities
        keep_indices = torch.multinomial(probabilities, num_patches_keep, replacement=False)

        # Gather the selected patches
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1)
        x = x[batch_indices, keep_indices]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)
            
        if return_indices:
            return x, keep_indices

        return x
