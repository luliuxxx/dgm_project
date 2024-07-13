import torch
from configs import MODEL_STORE
from utils.data import get_data_loader
import torchvision.transforms as transforms

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CHECKPOINT_DIR = Path(__file__).parent / "logs" / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"

def load_model(config_name, checkpoint_path):
    # Initialize the model
    model = MODEL_STORE[config_name]

    # Load the checkpoint
    full_checkpoint_path = f"{str(CHECKPOINT_DIR)}/{checkpoint_path}"
    checkpoint = torch.load(full_checkpoint_path)

    # Load state_dict into the model
    model.load_state_dict(checkpoint)

    # Set model to evaluation mode if you're using it for inference
    model.eval()

    return model


def generate_samples_for_label(model, label, num_samples=8):
    if label is None:
        return model.generate(num_samples)

    labels = list(label) * num_samples
    labels = torch.tensor(labels).reshape(num_samples, 1)

    # get samples
    samples = model.generate(num_samples, labels)
    return samples


def load_samples_for_label(dataloader, label_to_filter=None, num_samples=8):
    images = []
    for data in dataloader:
        imgs, labels = data
        for img, label in zip(imgs, labels):
            if label_to_filter is None or label_to_filter[0] in label:
                images.append(img)
                if len(images) == num_samples:
                    return torch.stack(images)
    return torch.stack(images)

def plot_n_save_samples(generated_images, real_images, title, num_samples, filename):
    # Plot images
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, num_samples * 3))
    fig.suptitle(title, fontsize=16)
    
    for i in range(num_samples):
        # Plot generated images
        img_gen = generated_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(img_gen)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title("Generated")

        # Plot real images
        img_real = real_images[i].detach().cpu().numpy().transpose(1, 2, 0)
        axes[i, 1].imshow(img_real)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title("Real")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.show()
    
    # Save the plot
    filepath = RESULTS_DIR / filename
    fig.savefig(filepath)
    print(f"Plot saved to {filepath}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parse arguments
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default="CVAE", choices=["CVAE"])
    parser.add_argument('--data_flag', type=str, default='pathmnist', choices=['pathmnist', 'breastmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'organmnist_axial', 'organmnist_coronal', 'organmnist_sagittal'])
    parser.add_argument("--label_to_binary", type=str)
    parser.add_argument("--num_samples", "--n", type=int, default=8)
    parser.add_argument("--ckpt_name", type=str)
    parser.add_argument("--use_config", type=str, default="CVAE_RGB")

    args = parser.parse_args()

    # load model
    model = load_model(args.use_config, args.ckpt_name)

    # load data
    _, val_loader, _, params = get_data_loader(args)
    label_dict = params["label_dict"]

    
    # generate and load images per label, then plot
    num_samples = args.num_samples
    for label, label_name in label_dict.items():
        label = np.array([int(label)])
        real_imgs =  load_samples_for_label(val_loader, label, num_samples)
        generated_images = generate_samples_for_label(model, label, num_samples)

        # un-normalize to range 0 - 1
        inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-1], std=[2])  # Equivalent to (x + 1) / 2
        ])
        real_imgs = inverse_transform(real_imgs)
        generated_images = inverse_transform(generated_images)
        print(f"REAL IMGS: {torch.min(real_imgs)} - {torch.max(real_imgs)}")
        print(f"REAL IMGS: {torch.min(generated_images)} - {torch.max(generated_images)}")

        # plot and save
        title = f"{args.model_name} with {args.data_flag} conditioned on {label_name}" if args.label_to_binary else f"{args.model_name} on {args.data_flag}" 
        filename = f"{args.model_name}_{args.use_config}_{args.data_flag}_{label_name}"
        plot_n_save_samples(generated_images, real_imgs, title, num_samples, filename)


