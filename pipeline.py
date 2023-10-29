import torch, torchvision
import os 
from typing import Union
from argparse import ArgumentParser
from torch.nn import Transformer
from torchvision.models import ResNet, VisionTransformer


def extract_params() -> dict[str, str]:
    """
    Extracts decoder hyperparameters provided
    in TXT file.
    """

    params = {}
    with open('decoder_hyperparams.txt', 'r') as params_file:
        for line in params_file.readlines():
            param, value = line.strip().split('=')
            params[param] = value
    
    return params


def craete_model(encoder : Union[ResNet, VisionTransformer], params : dict[str, str]) -> Transformer:
    """
    Combines passed encoder together with a decoder,
    initiated with parameters dictionary.
    """
    
    # All the hyperparameters should only be applicable to the decoder
    model = torch.nn.Transformer(d_model=int(params['d_model']),
                                 nhead=int(params['nhead']),
                                 num_decoder_layers=int(params['num_decoder_l']),
                                 dim_feedforward=int(params['dim_ff']),
                                 dropout=float(params['dropout']),
                                 activation=params['activation'],
                                 layer_norm_eps=float(params['layer_norm_eps']),
                                 custom_encoder=encoder)

    # TODO: Attach last layer to the decoder based on our vocabulary
    return model  


def main() -> Union[ResNet, VisionTransformer]:
    parser = ArgumentParser()
    parser.add_argument('--architecture', type=str, default='resnet-enc', help="Type in 1 of following: 'resnet-enc' or 'vit-enc'")
    args = parser.parse_args()

    architecture = args.architecture

    assert architecture in ('resnet-enc', 'vit-enc'), f"Can't recognize architecture: {architecture}"
    
    model = None
    params = extract_params()

    # Base ViT, 16x16 patches, no pre-trained weights
    if architecture == 'vit-enc':
        encoder = torchvision.models.vit_b_16()
        encoder.heads = torch.nn.Identity() 
        model = craete_model(encoder, params)

        return model

    # 34-layer plain ResNet, no pre-trained weights
    elif architecture == 'resnet-enc':
        encoder = torchvision.models.resnet34()  # possibly 18 layers?
        encoder.fc = torch.nn.Linear(in_features=512, out_features= int(params['d_model']))
        model = craete_model(encoder, params)

        return model


if __name__ == '__main__':
    model = main()
    print(model)