from modules.vae import VAE
from modules.vqvae import VQVAE
from modules.cvae import CVAE

class Config:
    """
    Configuration class to set attributes based on given keyword arguments.
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

CVAE_RGB_S = Config(
    model_name = "cvae",
    input_channels = 3,
    output_channels = 3,
    latent_channels = 128,
    hidden_channels = [8, 16],
    n_classes = 2,
    use_classes = True
)
CVAE_RGB = Config(
    model_name = "cvae",
    input_channels = 3,
    output_channels = 3,
    latent_channels = 256,
    intermediate_dims = 2,
    n_classes = 2,
    use_classes = True
)
CVAE_BW = Config(
    model_name = "cvae",
    input_channels = 2,
    output_channels = 2,
    latent_channels = 64,
    hidden_channels = [32, 64, 128, 256],
    n_classes = 2,
    use_classes = True
)

MODEL_STORE = {
    "CVAE_RGB": CVAE(CVAE_RGB),
    #"CVAE_BW": CVAE(CVAE_BW),
    "VQVAE": VQVAE(Config(model_name="vqvae"))
}
