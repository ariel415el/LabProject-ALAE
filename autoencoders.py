import numpy as np
import torch
import torch.nn as nn
from utils import plot_training
import torch.nn.functional as F


class BiGanMLP(nn.Module):
    def __init__(self, input_shape, hidden_dim=1024):
        super(BiGanMLP, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)
        self.droput = nn.BatchNorm2d

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class EncoderDecoder(object):
    def __init__(self, data_dim, latent_dim):
        self.training_dir = None
        self.name = "AbstractEncoderDecoder"
        self.data_dimdata_dim= data_dim
        self.latent_dim = latent_dim

    def __str__(self):
        return self.name

    def learn_encoder_decoder(self, data):
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

    def set_training_dir(self, training_dir):
        self.training_dir = training_dir

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
            self.E = torch.nn.Linear(data_dim, latent_dim)
            self.D = torch.nn.Linear(latent_dim, data_dim)
        else:
            raise Exception("Mode no supported")

    def learn_encoder_decoder(self, data):
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

        if self.training_dir:
            plot_training(losses, ["reconstruction_loss"], self.training_dir, self.name)

    def encode(self, zero_mean_data):
        with torch.no_grad():
            return self.E(torch.from_numpy(zero_mean_data).float()).numpy()

    def decode(self, encodings):
        with torch.no_grad():
            return self.D(torch.from_numpy(encodings).float()).numpy()


class ALAE(EncoderDecoder):
    def __init__(self, data_dim, latent_dim, z_dim=32, optimization_steps=1000, lr=0.002, batch_size=64, mode="Linear"):
        super(ALAE, self).__init__(data_dim, latent_dim)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.name = f"ALAE_z_dim[{z_dim}]_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]"

        if mode == "Linear":
            self.F = torch.nn.Linear(self.z_dim, latent_dim)
            self.G = torch.nn.Linear(latent_dim, data_dim)
            self.E = torch.nn.Linear(data_dim, latent_dim)
            self.D = torch.nn.Linear(latent_dim, 1)
        else:
            raise Exception("Mode no supported")

    def learn_encoder_decoder(self, data):
        ED_optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.D.parameters()), lr=self.lr, betas=(0.0, 0.99))
        FG_optimizer = torch.optim.Adam(list(self.F.parameters()) + list(self.G.parameters()), lr=self.lr, betas=(0.0, 0.99))
        EG_optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.G.parameters()), lr=self.lr, betas=(0.0, 0.99))

        softplus = F.softplus
        mse = F.mse_loss
        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        losses = [[], [], []]

        for s in range(self.optimization_steps):
            # Step I. Update E, and D
            ED_optimizer.zero_grad()
            batch_real_data = X[torch.randint(data.shape[0], (self.batch_size,))]
            batch_latent_vectors = torch.tensor(np.random.normal(0,1,size=(self.batch_size, self.z_dim)), requires_grad=False, dtype=torch.float32)
            L_adv_ED = softplus(self.D(self.E(self.G(self.F(batch_latent_vectors))))).mean() + softplus(-self.D(self.E(batch_real_data))).mean()
                        # TODO: + R1 gradient regularization as in paper
            L_adv_ED.backward()
            ED_optimizer.step()

            # Step II. Update F, and G
            FG_optimizer.zero_grad()
            batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=False, dtype=torch.float32)
            L_adv_FG = softplus(-self.D(self.E(self.G(self.F(batch_latent_vectors))))).mean()
            L_adv_FG.backward()
            FG_optimizer.step()

            # Step III. Update E, and G
            EG_optimizer.zero_grad()
            batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=False, dtype=torch.float32)
            L_err_EG = mse(self.F(batch_latent_vectors) , self.E(self.G(self.F(batch_latent_vectors))))
            L_err_EG.backward()
            EG_optimizer.step()

            losses[0] += [L_adv_ED.item()]
            losses[1] += [L_adv_FG.item()]
            losses[2] += [L_err_EG.item()]

        # plot training
        if self.training_dir:
            plot_training(losses, ["ED_loss", "FG_loss", 'EG_loss'], self.training_dir, self.name)

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
            self.G = nn.Linear(latent_dim, data_dim)
            self.E = nn.Linear(data_dim, latent_dim)
            # self.D = nn.Linear(data_dim, 1)
            self.D = nn.Sequential(nn.Linear(data_dim, 1), nn.Sigmoid())

        else:
            raise Exception("Mode no supported")

        self.BCE_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def learn_encoder_decoder(self, data):
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
        if self.training_dir:
            plot_training(losses, ["g-loss", "D-real", 'd-fake', 'z-reconstruction'], self.training_dir, self.name)

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
