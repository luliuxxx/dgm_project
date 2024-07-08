import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_channels = config.input_channels if hasattr(config, "input_channels") else 3
        self.class_size = config.class_size if hasattr(config, "class_size") else 3
        self.latent_dim = config.latent_channels if hasattr(config, "latent_channels") else 64
        self.hidden_dim = config.hidden_channels[0] if hasattr(config, "hidden_channels") else 128

        # Encoder layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.class_size, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(self.hidden_dim * 3 * 3, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim * 3 * 3, self.latent_dim)

        # Decoder layers
        self.decoder_fc = nn.Linear(self.latent_dim + self.class_size, self.hidden_dim * 3 * 3)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=4, stride=2, padding=1),  # (batch, 64, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (batch, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.input_channels, kernel_size=5, stride=2, padding=0, output_padding=1),  # (batch, input_channels, 32, 32)
            nn.Sigmoid()        )

    def one_hot(self, labels, class_size):
        targets = torch.zeros(labels.size(0), class_size)
        for i, label in enumerate(labels):
            targets[i, label] = 1
        return targets.to(labels.device)

    def encode(self, x, labels):
        one_hot_labels = self.one_hot(labels, self.class_size)
        one_hot_labels = one_hot_labels.unsqueeze(2).unsqueeze(3)
        one_hot_labels = one_hot_labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, one_hot_labels), dim=1)
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        one_hot_labels = self.one_hot(labels, self.class_size)
        z = torch.cat((z, one_hot_labels), dim=1)
        x = self.decoder_fc(z)
        x = x.view(x.size(0), self.hidden_dim, 3, 3)
        x = self.decoder_conv(x)
        return x

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, labels)
        loss = self.compute_loss(x, x_hat, mu, logvar)
        return loss, x_hat

    def compute_loss(self, x, x_hat, mu, logvar):
        mse = F.mse_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld

