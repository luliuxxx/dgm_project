import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_channels = config.input_channels
        self.output_channels = config.output_channels
        self.latent_channels = config.latent_channels
        self.hidden_channels = config.hidden_channels
        self.intermediate_dims = 7 # assuming 28x28 input, 4 downsample layers, then after encoder convolution, the output is B, C, 7, 7
        # Encoder definition
        encoder_modules = []
        for i in range(len(self.hidden_channels)):
            if i == 0:
                encoder_modules.append(nn.Conv2d(self.input_channels, self.hidden_channels[i], 4, 2, 1))
            else:
                encoder_modules.append(nn.Conv2d(self.hidden_channels[i-1], self.hidden_channels[i], 4, 2, 1))
            encoder_modules.append(nn.BatchNorm2d(self.hidden_channels[i]))
            encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        # FC layers for mu and logvar
        self.fc_mu = nn.Linear(self.hidden_channels[-1] * self.intermediate_dims * self.intermediate_dims, self.latent_channels)
        self.fc_logvar = nn.Linear(self.hidden_channels[-1] * self.intermediate_dims * self.intermediate_dims, self.latent_channels)
        self.proj_fc = nn.Linear(self.latent_channels, self.hidden_channels[-1] * self.intermediate_dims * self.intermediate_dims)
        # Decoder definition
        decoder_modules = []
        for i in range(len(self.hidden_channels) - 1, 0, -1):
            decoder_modules.append(nn.ConvTranspose2d(self.hidden_channels[i], self.hidden_channels[i-1], 4, 2, 1))
            decoder_modules.append(nn.BatchNorm2d(self.hidden_channels[i-1]))
            decoder_modules.append(nn.ReLU())
        decoder_modules.append(nn.ConvTranspose2d(self.hidden_channels[0], self.output_channels, 4, 2, 1))
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
       
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        proj_z = self.proj_fc(z)
        proj_z = proj_z.view(proj_z.size(0), -1, self.intermediate_dims, self.intermediate_dims)
        re_z = self.decoder(proj_z)
        return re_z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        loss = self.compute_loss(x, x_hat, mu, logvar)
        return loss, x_hat

    def compute_loss(self, x, x_hat, mu, logvar):
        mse = F.mse_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld
