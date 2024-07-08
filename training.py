# training script for the model
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from modules.vae import VAE
from modules.vqvae import VQVAE
from modules.cvae import CVAE
from utils.data import get_data_loader, get_parameters
from argparse import ArgumentParser
import wandb
import numpy as np
from tqdm import tqdm

class Config:
    """
    Configuration class to set attributes based on given keyword arguments.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer():
    def __init__(self, model, optimizer, device, args, use_labels=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.max_epochs = args.max_epochs
        self.log_path = './logs'
        self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        self.use_labels = use_labels

    def save_checkpoint(self, tag):
        torch.save(self.model.state_dict(), f'{self.log_path}/checkpoints/vae_{tag}.pt')
    
    def train_n_iters(self, batches_data):
        for images, labels in tqdm(batches_data):
            images = images.to(self.device)
            if self.use_labels:
                labels = labels.to(self.device)
                loss, x_hats = self.model(images, labels)
            else:
                loss, x_hats = self.model(images)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return None

    @torch.no_grad()
    def evaluate(self, train_loader,val_loader):
        self.model.eval()
        losses = {'train': [], 'val': []}
        for data in [train_loader, val_loader]:
            for batch_data in data:
                x, labels = batch_data
                x = x.to(self.device)
                if self.use_labels:
                    labels = labels.to(self.device)
                    loss, x_hats = self.model(x, labels)
                else:
                    loss, x_hats = self.model(x)
                losses['train' if data == train_loader else 'val'].append(loss.item())
        # compute mean loss
        mean_losses = {k: sum(v)/len(v) for k, v in losses.items()}
        print(f'Train Loss: {mean_losses["train"]}, Val Loss: {mean_losses["val"]}')
        self.scheduler.step(mean_losses['val'])
        lr = self.scheduler.get_last_lr()
        print(f'Learning rate: {lr}')
        if self.args.wandb:
            wandb_images_x_hat = wandb.Image(x_hats, caption='reconstructed')
            wandb_images_x = wandb.Image(x, caption='original')
            wandb.log({"original": wandb_images_x,"reconstructed": wandb_images_x_hat})
            wandb.log({"train loss": mean_losses['train'], "val loss": mean_losses['val']})
        self.model.train()
    
    
    def fit(self, train_loader, val_loader):
        self.model.train()
        for epoch in range(self.max_epochs):
            print(f'Epoch {epoch}')
            # start training
            self.train_n_iters(train_loader)
            if epoch % self.args.eval_freq == 0:
                self.evaluate(train_loader,val_loader)
            if epoch % self.args.ckpt_freq == 0 and epoch != 0:
                self.save_checkpoint(epoch)
        if self.args.wandb:
            wandb.finish()
        return None
    
    @torch.no_grad()
    def test(self, test_loader):
        self.model.eval()
        for batch_data in test_loader:
            x, labels = batch_data
            x = x.to(self.device)
            if self.use_labels:
                labels = labels.to(self.device)
                loss, x_hats = self.model(x, labels)
            else:
                loss, x_hats = self.model(x)
            loss, x_hat = self.model(x)
            print(f'Test Loss: {loss.item()}')
        return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parse arguments
    parser = ArgumentParser()

    # training arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--ckpt_freq', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate','--lr', type=float, default=1e-3)
    parser.add_argument('--wandb', type=int, default=0, choices=[0, 1])
    parser.add_argument('--state', type=str, default='train', choices=['train', 'test'])
    # parser.add_argument('--load_checkpoint', type=str, default='/home/lliu/dgm_project/logs/checkpoints/vae_200.pt') # TODO: load the latest checkpoint
    
    # data arguments, multiple for multi-modality
    parser.add_argument('--data_flag', nargs="+", type=str, default='pathmnist', choices=['pathmnist', 'breastmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'organmnist_axial', 'organmnist_coronal', 'organmnist_sagittal'])

    args = parser.parse_args()

    # load data
    train_loader, val_loader, test_loader, params = get_data_loader(args)

    # model configuration
    input_channels = params['n_channels'] 
    output_channels = input_channels
    latent_channels = 64
    hidden_channels = [32, 64, 128, 256]
    n_classes = params['n_classes']

    # vae_config = Config(input_channels = input_channels,
    #                     output_channels = output_channels,
    #                     latent_channels = latent_channels,
    #                     hidden_channels = hidden_channels)
    # model = VAE(vae_config).to(device)

    if n_classes == 1:
        vqvae_config = Config()
        model = VQVAE(vqvae_config).to(device)
        model_name = 'vqvae'
        use_labels = False
    else:
        cvae_config = Config()
        model = CVAE(cvae_config).to(device)
        model_name = 'cvae'
        use_labels = True
    print(f"Number of parameters: {get_parameters(model):,}")
    # import ipdb; ipdb.set_trace()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    trainer = Trainer(model, optimizer, device, args, use_labels)

    if args.state=='train' and args.wandb:
        wandb.init(
            project = 'dgm_project',
            config = {
                'model': model_name,
                'dataset': "+".join(args.data_flag),
            }
        )

    if args.state == 'train':
        trainer.fit(train_loader, val_loader)
    else:
        trainer.test(test_loader)
    return None


if __name__ == '__main__':
    main()
