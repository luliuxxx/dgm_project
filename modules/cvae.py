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

        # encoder layers
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv1 = nn.Conv2d(3 + self.class_size, 8, kernel_size=3, stride=1, padding="same")
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding="same")
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same")
        self.fc_mu = nn.Linear(32 * self.intermediate_dims * self.intermediate_dims, self.latent_channels)
        self.fc_logvar = nn.Linear(32 * self.intermediate_dims * self.intermediate_dims, self.latent_channels)

        # decoder layers
        self.fc_proj = nn.Linear(self.latent_channels + self.class_size, 32 * self.intermediate_dims * self.intermediate_dims)

        # convolutions
        self.convT1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, output_padding=0) #SAME PADDING
        self.convT2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, output_padding=1)
        self.convT3 = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, output_padding=0)


    def one_hot(self,labels, class_size):
        targets = torch.zeros(labels.size(0), class_size, device=labels.device)
        targets = targets.scatter_(1, labels, 1)
        return targets

    def encode(self, x, labels):
        one_hot_labels = self.one_hot(labels, self.class_size)
        one_hot_labels = one_hot_labels.unsqueeze(2).unsqueeze(3)
        one_hot_labels = one_hot_labels.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x, one_hot_labels), dim=1)
        # Conv layers 
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        mu = self.fc_mu(x) 
        mu = F.relu(mu)
        logvar = self.fc_logvar(x)
        logvar = F.relu(logvar) 

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        one_hot_labels = self.one_hot(labels, self.class_size)
        z = torch.cat((z, one_hot_labels), dim=1)
        # Fully connected layer to project z and labels back to feature map

        # FC layer
        z = self.fc_proj(z)
        z = F.relu(z)

        # Unflatten
        z = z.view(z.size(0), -1, self.intermediate_dims, self.intermediate_dims)

        # Conv layers
        z = self.convT1(z)
        z = F.relu(z)
        z = self.convT2(z)
        z = F.relu(z)
        z = self.convT3(z)

        # cut
        z = z[:, :, :-1, :-1]

        return z

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, labels)
        loss = self.compute_loss(x, x_hat, mu, logvar)
        return loss, x_hat

    def compute_loss(self, x, x_hat, mu, logvar):
        mse = F.mse_loss(x_hat, x, reduction="sum")
        # kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        kld = torch.sum(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1))
        total = mse + 0.3 * kld
        return total
 
    def generate(self, num_samples, labels):
        # Sample from the latent space
        z = torch.randn(num_samples, self.latent_channels).to(next(self.parameters()).device) * 3
        labels = torch.tensor(labels).to(next(self.parameters()).device)
        # Decode the samples
        generated_images = self.decode(z, labels)
        return generated_images
