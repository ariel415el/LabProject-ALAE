import numpy as np
import torch
import torch.nn as nn
from utils import plot_training
import torch.nn.functional as F
from time import time
import os

class BiGanMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, activation=torch.nn.ReLU()):
        super(BiGanMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # x = self.batch_norm(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


class PixelNormLayer(nn.Module):
    """Taken from https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/CustomLayers.py"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class EncoderDecoder(object):
    def __init__(self, data_dim, latent_dim):
        self.name = "AbstractEncoderDecoder"
        self.data_dim =  data_dim
        self.latent_dim = latent_dim

    def __str__(self):
        return self.name

    def learn_encoder_decoder(self, data, plot_dir=None):
        """
        Learns the encoder and decoder transformations from data
        :param laten_dim: size of the latent space of the desired encodings
        """
        raise NotImplementedError

    def encode(self, zero_mean_data):
        raise NotImplementedError

    def decode(self, zero_mean_data):
        raise NotImplementedError

    def get_reconstuction_loss(self, zero_mean_data):
        return np.linalg.norm(zero_mean_data - self.decode(self.encode(zero_mean_data)), ord=2) / zero_mean_data.shape[0]


class IdentityAutoEncoder(EncoderDecoder):
    def __init__(self, data_dim, latent_dim):
        super(IdentityAutoEncoder, self).__init__(data_dim, latent_dim)
        assert(latent_dim is None or data_dim == latent_dim)
        self.name = "OriginalData"

    def learn_encoder_decoder(self, data, plot_dir=None):
        pass

    def encode(self, zero_mean_data):
        return zero_mean_data

    def decode(self, zero_mean_data):
        return zero_mean_data


class VanilaAE(EncoderDecoder):
    """
    SGD minimzation of reconstruction loss ||X - XED|| where E and D are any
    d x laten_dim and laten_dim x d matrices.
    """
    def __init__(self, data_dim, latent_dim, optimization_steps=1000, lr=0.001, batch_size=64,
                 mode='Linear', metric='l2'):
        super(VanilaAE, self).__init__(data_dim, latent_dim)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.name = f"VanilaAE({metric})_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]"
        self.metric = F.l1_loss if metric == 'l1' else F.mse_loss
        if mode == "Linear":
            self.E = torch.nn.Linear(data_dim, latent_dim, bias=False)
            self.D = torch.nn.Linear(latent_dim, data_dim, bias=False)
        else:
            raise Exception("Mode no supported")

    def learn_encoder_decoder(self, data, plot_dir=None):
        start = time()
        print("\tLearning encoder decoder... ",end="")
        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.D.parameters()), lr=self.lr)

        losses = [[]]

        for s in range(self.optimization_steps):
            batch_X = X[torch.randint(X.shape[0], (self.batch_size,), dtype=torch.long)]
            reconstruct_data = self.D(self.E(batch_X))

            loss = self.metric(batch_X, reconstruct_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [loss.item()]

        if plot_dir:
            plot_training(losses, ["reconstruction_loss"], os.path.join(plot_dir, f"Learning-{self}.png"))

        print(f"Finished in {time() - start:.2f} sec")

    def encode(self, zero_mean_data):
        with torch.no_grad():
            return self.E(torch.from_numpy(zero_mean_data).float()).numpy()

    def decode(self, encodings):
        with torch.no_grad():
            return self.D(torch.from_numpy(encodings).float()).numpy()


from torch.utils.data import Dataset
class SimpleDataset(Dataset):
    def __init__(self, data_matrix):
        # super(SimpleDataset, self).__init__()
        self.data_matrix = data_matrix

    def __len__(self):
        return len(self.data_matrix)

    def __getitem__(self, idx):
        return self.data_matrix[idx]

    def get_data(self):
        return self.data_matrix


class ALAE(EncoderDecoder):
    def __init__(self, data_dim, latent_dim, z_dim=32, epochs=600, lr=0.002, batch_size=128,
                 g_penalty_coeff=10.0, mode="Linear"):
        super(ALAE, self).__init__(data_dim, latent_dim)
        self.epochs = epochs
        self.lr = lr
        self.z_dim = z_dim
        self.g_penalty_coeff = g_penalty_coeff
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = f"ALAE_z_dim[{z_dim}]_e[{epochs}]_lr[{lr}]_b[{batch_size}]"

        if mode == "Linear":
            self.F = torch.nn.Linear(self.z_dim, latent_dim, bias=False)
            self.G = torch.nn.Linear(latent_dim, data_dim, bias=False)
            self.E = torch.nn.Linear(data_dim, latent_dim, bias=False)
            self.D = torch.nn.Linear(latent_dim, 1, bias=False)
            # self.D = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

        elif "mnist":
            self.F = nn.Sequential(PixelNormLayer(), nn.Linear(self.z_dim, latent_dim), nn.LeakyReLU(0.2),
                                   # nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2),
                                   # nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2),
                                   # nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2),
                                   # nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2),
                                   nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2),
                                   nn.Linear(latent_dim, latent_dim), nn.LeakyReLU(0.2),
                                   nn.Linear(latent_dim, latent_dim))
            self.G = BiGanMLP(latent_dim, data_dim, 1024)
            self.E = nn.Sequential(BiGanMLP(data_dim, latent_dim, 1024, activation=nn.LeakyReLU(0.2)), nn.LeakyReLU(0.2))
            self.D = BiGanMLP(latent_dim, 1, 1024, activation=nn.LeakyReLU(0.2))
        else:
            raise Exception("Mode no supported")

    def learn_encoder_decoder(self, data, plot_dir=None):
        start = time()
        print("\tLearning encoder decoder... ",end="")
        dataset = SimpleDataset(data)
        kwargs = {'batch_size': self.batch_size}
        if self.device != "cpu":
            kwargs.update({'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True},
                          )

        train_loader = torch.utils.data.DataLoader(dataset, **kwargs)

        ED_optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.D.parameters()), lr=self.lr, betas=(0.0, 0.99))
        FG_optimizer = torch.optim.Adam(list(self.F.parameters()) + list(self.G.parameters()), lr=self.lr, betas=(0.0, 0.99))
        EG_optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.G.parameters()), lr=self.lr, betas=(0.0, 0.99))

        softplus = F.softplus
        mse = F.mse_loss
        # X = torch.tensor(data, requires_grad=True, )

        losses = [[], [], []]

        for epoch in range(self.epochs):
            for batch_idx, batch_real_data in enumerate(train_loader):
                batch_real_data = batch_real_data.requires_grad_(True).float()
                # Step I. Update E, and D:  Optimize the discriminator D(E( * )) to better differentiate between real x data
                # and data generated by G(F( * ))
                ED_optimizer.zero_grad()
                batch_latent_vectors = torch.tensor(np.random.normal(0,1,size=(self.batch_size, self.z_dim)), requires_grad=False, dtype=torch.float32)
                real_images_dicriminator_outputs = self.D(self.E(batch_real_data))
                L_adv_ED = softplus(self.D(self.E(self.G(self.F(batch_latent_vectors))))).mean() + softplus(-real_images_dicriminator_outputs).mean()

                # R1 gradient regularization as in paper
                real_grads = torch.autograd.grad(outputs=real_images_dicriminator_outputs, inputs=batch_real_data,
                                                 grad_outputs=torch.ones_like(real_images_dicriminator_outputs),
                                                 create_graph=True, retain_graph=True)[0]
                gradient_penalty = 0.5 * ((real_grads.norm(2, dim=1) - 1) ** 2).mean()
                L_adv_ED += gradient_penalty * self.g_penalty_coeff

                L_adv_ED.backward()
                ED_optimizer.step()

                # Step II. Update F, and G: Optimize the generator G(F( * )) to fool D(E ( * ))
                FG_optimizer.zero_grad()
                batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=False, dtype=torch.float32)
                L_adv_FG = softplus(-self.D(self.E(self.G(self.F(batch_latent_vectors))))).mean()
                L_adv_FG.backward()
                FG_optimizer.step()

                # Step III. Update E, and G: Optimize the reconstruction loss in the Latent space W
                EG_optimizer.zero_grad()
                batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=False, dtype=torch.float32)
                w_latent_vectors = self.F(batch_latent_vectors).detach()
                L_err_EG = mse(w_latent_vectors, self.E(self.G(w_latent_vectors)))
                L_err_EG.backward()
                EG_optimizer.step()

                losses[0] += [L_adv_ED.item()]
                losses[1] += [L_adv_FG.item()]
                losses[2] += [L_err_EG.item()]

            print(f"Epoch done {epoch}")
            # plot training
            if plot_dir:
                plot_training(losses, ["ED_loss", "FG_loss", 'EG_loss'], os.path.join(plot_dir, f"Learning-{self}.png"))

        print(f"Finished in {time() - start:.2f} sec")

    def encode(self, zero_mean_data):
        with torch.no_grad():
            return self.E(torch.from_numpy(zero_mean_data).float()).numpy()

    def decode(self, encodings):
        with torch.no_grad():
            return self.G(torch.from_numpy(encodings).float()).numpy()


class LatentRegressor(EncoderDecoder):
    """
    This architecture is suggested in the BiGan paper "Adverserial Feature Learning": https://arxiv.org/abs/1605.09782
    It is basicly a GAN architecture where an encoder is trained (in parrallel to the gan or afterwards) to minimize
    te reconstruction loss in the latent space
    """
    def __init__(self, data_dim, latent_dim, optimization_steps=1000, lr=0.002, batch_size=64,
                 mode="Linear", regressor_training='joint'):
        super(LatentRegressor, self).__init__(data_dim, latent_dim)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.regressor_training = regressor_training
        self.name = f"LR_dim_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]"
        if regressor_training == "joint":
            self.name = "J" + self.name

        if mode == "Linear":
            self.G = nn.Linear(latent_dim, data_dim, bias=False)
            self.E = nn.Linear(data_dim, latent_dim, bias=False)
            # self.D = nn.Linear(data_dim, 1)
            self.D = nn.Sequential(nn.Linear(data_dim, 1, bias=False), nn.Sigmoid())

        else:
            raise Exception("Mode no supported")

        self.BCE_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def learn_encoder_decoder(self, data, plot_dir=None):
        start = time()
        print("\tLearning encoder decoder... ",end="")
        # Optimizers
        optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_E = torch.optim.Adam(self.E.parameters(), lr=self.lr, betas=(0.5, 0.999))

        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        losses = [[], [], [], []]

        for s in range(self.optimization_steps):
            # Adversarial ground truths
            ones = torch.ones((self.batch_size, 1), dtype=torch.float32, requires_grad=False)
            zeros = torch.zeros((self.batch_size, 1), dtype=torch.float32, requires_grad=False)
            batch_real_data = X[torch.randint(data.shape[0], (self.batch_size,))]

            #  Train Generator #
            optimizer_G.zero_grad()
            z = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.latent_dim)), dtype=torch.float32, requires_grad=False)
            generated_data = self.G(z)
            g_loss = self.BCE_loss(self.D(generated_data), ones)
            g_loss.backward()
            optimizer_G.step()

            # Train discriminator #
            optimizer_D.zero_grad()
            fake_loss = self.BCE_loss(self.D(generated_data.detach()), zeros) # detach so that no gradient will be computed for G
            real_loss = self.BCE_loss(self.D(batch_real_data), ones)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            losses[0] += [g_loss.item()]
            losses[1] += [real_loss.item()]
            losses[2] += [fake_loss.item()]

            if self.regressor_training == 'joint':
                losses[3] +=[self.regress_encoder(optimizer_E, z)]

        if self.regressor_training != 'joint':
            for s in range(self.optimization_steps):
                z = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.latent_dim)), dtype=torch.float32, requires_grad=False)
                losses[3] += [self.regress_encoder(optimizer_E, z)]

        # plot training
        if plot_dir:
            plot_training(losses, ["g-loss", "D-real", 'd-fake', 'z-reconstruction'], os.path.join(plot_dir, f"Learning-{self}.png"))

        print(f"Finished in {time() - start:.2f} sec")

    def regress_encoder(self, optimizer, latent_batch):
        optimizer.zero_grad()
        # TODO : is this what they meant by sigmoid cross entropy loss in the BiGan paper
        loss = self.BCE_loss(self.sigmoid(self.E(self.G(latent_batch).detach())), latent_batch)
        loss.backward()
        optimizer.step()

        return loss

    def encode(self, zero_mean_data):
        with torch.no_grad():
            return self.E(torch.from_numpy(zero_mean_data).float()).numpy()

    def decode(self, encodings):
        with torch.no_grad():
            return self.G(torch.from_numpy(encodings).float()).numpy()
