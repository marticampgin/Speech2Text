import torch
import timm

from torch import nn
from timm.models.layers import to_2tuple
from typing import Tuple
from decoder import DecoderRNN


# Override the timm package to relax the input shape constraint 
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Img2SeqModel(nn.Module):
    def __init__(self,
                 encoder_dim : int,
                 decoder_dim : int,
                 embed_size : int,
                 attention_dim : int,
                 vocab_size : int,
                 img_shape : Tuple[int]) -> None:
        
        super().__init__()


        # ------------ INITIALIZING ENCODER ------------

        # Override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # Load pre-trained encoder (data-efficient image Transformer trained with CNN knowledge distillation)
        self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=True)
        
        # Extract / init variables
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]

        self.tstride = 10
        self.fstride = 10
        _, self.h, self.w = img_shape

        # Calculate num. of patches given current img. dimensions & patch overlap
        f_dim, t_dim = self.get_shape(self.fstride, self.tstride, self.h, self.w)

        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        # Linear projection layer 
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(self.fstride , self.tstride))

        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))  # Do they actually sum them or take avg?
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        # Get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24)
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim)
        new_pos_embed = new_pos_embed.transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)

        # Apply bilinear interpolation of weights
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')

        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

        # Flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)

        # Concatenate the above positional embedding with the cls token of the deit model.
        self.v.pos_embed = nn.Parameter(new_pos_embed)

        # ------------ INITIALIZING DECODER ------------
        self.decoder = DecoderRNN(embed_size=embed_size,
                                  vocab_size=vocab_size,
                                  attention_dim=attention_dim,
                                  encoder_dim=encoder_dim,
                                  decoder_dim=decoder_dim)


    def encode(self, src):
        # Embed spectrograms
        src = self.v.patch_embed(src)
        src = src + self.v.pos_embed
        src = self.v.pos_drop(src)
        # Run through transformer encoder blocks & normalize
        for blk in self.v.blocks:
            src = blk(src)
        
        return  self.v.norm(src)
    

    def forward(self, src : torch.tensor, 
                tgt : torch.tensor) -> torch.Tensor:
        # Encode
        features = self.encode(src)

        # Decode
        outputs = self.decoder(features, tgt)

        return outputs


    def get_shape(self, fstride, tstride, height, width):
        test_input = torch.randn(1, 1, height, width)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim