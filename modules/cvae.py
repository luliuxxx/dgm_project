import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channels = config.input_channels if hasattr(config, "input_channels") else 3
        self.class_size = config.class_size if hasattr(config, "class_size") else 3
        self.output_channels = config.output_channels if hasattr(config, 'output_channels') else 3
        self.latent_channels = config.latent_channels if hasattr(config, 'latent_channels') else 64
        self.hidden_channels = config.hidden_channels if hasattr(config, 'hidden_channels') else [32, 64, 128, 256]
        self.intermediate_dims = 4 # assuming 28x28 input, 4 downsample layers, then after encoder convolution, the output is B, C, 7, 7

        encoder_modules = []
        for i in range(len(self.hidden_channels)):
            if i == 0:
                encoder_modules.append(nn.Conv2d(self.input_channels + self.class_size, self.hidden_channels[i], 3, 1, 1))
            else:
                encoder_modules.append(nn.Conv2d(self.hidden_channels[i-1], self.hidden_channels[i], 3, 2, 1))
            encoder_modules.append(nn.BatchNorm2d(self.hidden_channels[i]))
            encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        # FC layers for mu and logvar
        self.fc_mu = nn.Linear(self.hidden_channels[-1] * self.intermediate_dims * self.intermediate_dims, self.latent_channels)
        self.fc_logvar = nn.Linear(self.hidden_channels[-1] * self.intermediate_dims * self.intermediate_dims, self.latent_channels)
        self.proj_fc = nn.Linear(self.latent_channels + self.class_size, self.hidden_channels[-1] * self.intermediate_dims * self.intermediate_dims)
        # Decoder definition
        decoder_modules = []
        for i in range(len(self.hidden_channels) - 1, 0, -1):
            decoder_modules.append(nn.ConvTranspose2d(self.hidden_channels[i], self.hidden_channels[i-1], 3, 2, 1))
            decoder_modules.append(nn.BatchNorm2d(self.hidden_channels[i-1]))
            decoder_modules.append(nn.ReLU())
        decoder_modules.append(nn.ConvTranspose2d(self.hidden_channels[0], self.output_channels, 4, 1, 0))
        self.decoder = nn.Sequential(*decoder_modules)

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
        x = self.encoder(x)
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
        proj_z = self.proj_fc(z)
        proj_z = proj_z.view(proj_z.size(0), -1, self.intermediate_dims, self.intermediate_dims)
        re_z = self.decoder(proj_z)
        return re_z

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

    def generate(self, num_samples, labels):
        # Sample from the latent space
        z = torch.randn(num_samples, self.latent_channels).to(next(self.parameters()).device)
        labels = torch.tensor(labels).to(next(self.parameters()).device)
        # Decode the samples
        generated_images = self.decode(z, labels)
        return generated_images
