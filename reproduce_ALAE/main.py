import torch
from ALAE import ALAE
from datasets import get_dataset, get_dataloader
import numpy as np
from StyleGan import StyleGan

OUTPUT_DIR= 'training_dir_3_opts'
LATENT_SPACE_SIZE = 50
NUMED_BUG_IMAGES=32
EPOCHS=100
# DATASET = "mnist"
DATASET = "lfw"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def trainGAN():
    # Create datasets
    train_dataset, test_dataset, img_dim = get_dataset("../data", DATASET)

    # Create model
    hp = {'lr': 0.002, "batch_size": 128, 'mapping_layers':6}
    # model = ALAE(z_dim=LATENT_SPACE_SIZE, w_dim=LATENT_SPACE_SIZE, image_dim=img_dim, hyper_parameters=hp, device=device)
    model = StyleGan(z_dim=LATENT_SPACE_SIZE, w_dim=LATENT_SPACE_SIZE, image_dim=img_dim, hyper_parameters=hp, device=device)

    test_dataloader = get_dataloader(test_dataset, batch_size=NUMED_BUG_IMAGES, resize=None, device=device)
    test_samples_z = torch.tensor(np.random.RandomState(3456).randn(NUMED_BUG_IMAGES, LATENT_SPACE_SIZE),
                                  dtype=torch.float32).to(device)
    test_samples = next(iter(test_dataloader))

    # model.train(train_dataset, (test_samples_z, test_samples), EPOCHS, OUTPUT_DIR)
    model.train(train_dataset, (test_samples_z, test_samples), OUTPUT_DIR)


if __name__ == '__main__':
    trainGAN()