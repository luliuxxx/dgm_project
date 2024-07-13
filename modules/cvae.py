import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.class_size = config.n_classes
        self.latent_channels = config.latent_channels
        self.intermediate_dims = config.intermediate_dims

        # Encoder: 3 input channels (RGB) + class_size for labels
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + self.class_size, 8, kernel_size=3, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
        )

        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Linear(32 * self.intermediate_dims * self.intermediate_dims, self.latent_channels)
        self.fc_logvar = nn.Linear(32 * self.intermediate_dims * self.intermediate_dims, self.latent_channels)

        # Fully connected layer to project z and labels back to feature map
        self.proj_fc = nn.Linear(self.latent_channels + self.class_size, 32 * self.intermediate_dims * self.intermediate_dims)

        # Decoder: in_channels = 64, out_channels = 3 (RGB)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, output_padding=1),
            nn.Tanh()  # Assuming output is normalized between -1 and 1
        )

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
        mse = F.mse_loss(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld

    def generate(self, num_samples, labels):
        # Sample from the latent space
        z = torch.randn(num_samples, self.latent_channels).to(next(self.parameters()).device)
        labels = torch.tensor(labels).to(next(self.parameters()).device)
        # Decode the samples
        generated_images = self.decode(z, labels)
        return generated_images
