import numpy as np
import torch
from utils import plot_training
from tqdm import tqdm


class EncoderDecoder(object):
    def __init__(self, data_dim, latent_dim, outputs_dir=None):
        self.outputs_dir = outputs_dir
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


class VanilaAE(EncoderDecoder):
    """
    SGD minimzation of reconstruction loss ||X - XED|| where E and D are any
    d x laten_dim and laten_dim x d matrices.
    """
    def __init__(self, data_dim, latent_dim, training_dir, optimization_steps=1000, lr=0.001, batch_size=64, mode='Linear'):
        super(VanilaAE, self).__init__(data_dim, latent_dim, training_dir)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size
        self.name = f"VanilaAE_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]"
        if mode == "Linear":
            self.E = torch.nn.Linear(data_dim, latent_dim)
            self.D = torch.nn.Linear(latent_dim, data_dim)
        else:
            raise Exception("Mode no supported")

    def learn_encoder_decoder(self, data):
        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.D.parameters()), lr=self.lr)

        losses = [[]]

        for s in tqdm(range(self.optimization_steps)):
            batch_X = X[torch.randint(X.shape[0], (self.batch_size,), dtype=torch.long)]
            reconstruct_data = self.D(self.E(batch_X))
            loss = torch.nn.functional.mse_loss(batch_X, reconstruct_data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [loss.item()]

        if self.outputs_dir:
            plot_training(losses, ["reconstruction_loss"], self.outputs_dir, self.name)

    def encode(self, zero_mean_data):
        with torch.no_grad():
            return self.E(torch.from_numpy(zero_mean_data).float()).numpy()

    def decode(self, encodings):
        with torch.no_grad():
            return self.D(torch.from_numpy(encodings).float()).numpy()


class ALAE(EncoderDecoder):
    def __init__(self, data_dim, latent_dim, training_dir, z_dim=32, optimization_steps=1000, lr=0.002, batch_size=64, mode="Linear"):
        super(ALAE, self).__init__(data_dim, latent_dim, training_dir)
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

        softplus = torch.nn.functional.softplus
        mse = torch.nn.functional.mse_loss
        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        losses = [[], [], []]

        for s in tqdm(range(self.optimization_steps)):
            # Step I. Update E, and D
            ED_optimizer.zero_grad()
            batch_real_data = X[torch.randint(data.shape[0], (self.batch_size,))]
            batch_latent_vectors = torch.tensor(np.random.normal(0,1,size=(self.batch_size, self.z_dim)), requires_grad=True, dtype=torch.float32)
            L_adv_ED = softplus(self.D(self.E(self.G(self.F(batch_latent_vectors))))).mean() + softplus(-self.D(self.E(batch_real_data))).mean()
                        # TODO: + R1 gradient regularization as in paper
            L_adv_ED.backward()
            ED_optimizer.step()

            # Step II. Update F, and G
            FG_optimizer.zero_grad()
            batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=True, dtype=torch.float32)
            L_adv_FG = softplus(-self.D(self.E(self.G(self.F(batch_latent_vectors))))).mean()
            L_adv_FG.backward()
            FG_optimizer.step()

            # Step III. Update E, and G
            EG_optimizer.zero_grad()
            batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=True, dtype=torch.float32)
            L_err_EG = mse(self.F(batch_latent_vectors) , self.E(self.G(self.F(batch_latent_vectors))))
            L_err_EG.backward()
            EG_optimizer.step()

            losses[0] += [L_adv_ED.item()]
            losses[1] += [L_adv_FG.item()]
            losses[2] += [L_err_EG.item()]

        # plot training
        if self.outputs_dir:
            plot_training(losses, ["ED_loss", "FG_loss", 'EG_loss'], self.outputs_dir, self.name)

    def encode(self, zero_mean_data):
        with torch.no_grad():
            return self.E(torch.from_numpy(zero_mean_data).float()).numpy()

    def decode(self, encodings):
        with torch.no_grad():
            return self.G(torch.from_numpy(encodings).float()).numpy()