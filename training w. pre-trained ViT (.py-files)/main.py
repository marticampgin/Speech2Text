import matplotlib.pyplot as plt
import matplotlib
import torch
import tqdm

from torchvision import transforms
from dataset import CustomSpectroDataset, CollateFunctor
from torch.utils.data import DataLoader
from encoder_model import Img2SeqModel
from torch import nn
from argparse import ArgumentParser


def train(model, iterator, enc_optim, dec_optim, criterion, vocab_size, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm.tqdm(iterator):
        img = batch['img'].to(device)
        target_seq = batch['seq'].to(device)

        outputs, _ = model(img, target_seq)
        targets = target_seq[:, 1:]
        
        loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
        loss.backward()

        enc_optim.step()
        dec_optim.step()

        total_loss += loss.detach().item()
  
    return total_loss / len(iterator)


@torch.no_grad()
def evaluate(model, iterator, evaluations, dataset, device):
    model.eval()
    batch = next(iter(iterator))
    
    # Evaluatuing on n spectrograms (n <= 15)
    for i in range(evaluations):
        img = batch['img'][i].unsqueeze(0).to(device)
        target_seq = batch['seq'][i]

        features = model.encode(img)
        caps, _ = model.decoder.generate_caption(features, int2str=dataset.int2str, str2int=dataset.str2int, max_len=16)
        caption = ' '.join(caps)
        target = [dataset.int2str[i.item()] for i in target_seq.detach().cpu()]

        print(f'Generated caption: {caption}')
        print(f'Target caption: {" ".join(t for t in target)}', end='\n-----------------------\n\n')


def main():
    parser = ArgumentParser()
    parser.add_argument("--csv_dataset", default='dataset.csv')
    parser.add_argument("--wav2spec", default=False, help="Set to true if you have .wav recording and want to convert to spectrograms")
    parser.add_argument('--wav_folder', default='wav_recordings', help='folder w. wav recordings')
    parser.add_argument('--spec_folder', default='spectrograms', help='folder w. spectrograms')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--encoder_dim', default=768, type=int)
    parser.add_argument('--decoder_dim', default=768, type=int)
    parser.add_argument('--embed_size', default=300, type=int)
    parser.add_argument('--attention_dim', default=384)
    parser.add_argument('--evaluations', default=10, help='num. of spectrograms to evaluate (max 15)', type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--enc_lr', default=1e-4, type=float)
    parser.add_argument('--dec_lr', default=1e-5, type=float)
    args = parser.parse_args()
    
    # Loading dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CustomSpectroDataset(csv_data=args.csv_dataset,
                                   produce_spectr=args.wav2spec,
                                   transform=transform,
                                   recordings_folder=args.wav_folder,
                                   spectr_folder=args.spec_folder)

    # Train/test splits
    seed = 77
    gen = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=gen)


    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  collate_fn=CollateFunctor(pad_idx=dataset.str2int['[PAD]']))

    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=True,
                                 collate_fn=CollateFunctor(pad_idx=dataset.str2int['[PAD]']))


    # Model and its hyperparameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = dataset.get_vocab_size()
    img_shape = (1, 128, 1024)  # image shape of largest spectrogram

    model = Img2SeqModel(encoder_dim=args.encoder_dim,
                         decoder_dim=args.decoder_dim,
                         embed_size=args.embed_size,
                         attention_dim=args.attention_dim,
                         vocab_size=vocab_size,
                         img_shape=img_shape)

    model = model.to(device)

    evaluations = args.evaluations  # how many spectrograms to evaluate (max. = 15)
    epochs = args.epochs
    enc_lr = args.enc_lr
    dec_lr = args.dec_lr
    enc_optim = torch.optim.AdamW(model.v.parameters(), lr=enc_lr)
    dec_optim = torch.optim.AdamW(model.decoder.parameters(), lr=dec_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.str2int['[PAD]'])


    for epoch in range(epochs):
        epoch_loss = train(model, train_dataloader, enc_optim, dec_optim, criterion, vocab_size, device)
        print(f'Epoch: {epoch + 1}, Loss: {epoch_loss}', end='\n\n')
        evaluate(model, test_dataloader, evaluations, dataset, device)
    

if __name__ == "__main__":
    main()
