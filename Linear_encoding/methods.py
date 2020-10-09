import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
COLORS =['r', 'g', 'b']


def plot_training(losses, names, plots_dir, plot_name):
    fig = plt.figure(figsize=(10, 6))
    for i, (loss_list, name) in enumerate(zip(losses, names)):
        ax = fig.add_subplot(1, len(losses), 1 + i)
        ax.set_title(name)
        ax.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(os.path.join(plots_dir, plot_name + ".png"))
    plt.clf()

class LinearAutoEncoder(object):
    def __init__(self, laten_dim, name, outputs_dir=None):
        self.name = name
        self.laten_dim = laten_dim
        self.outputs_dir = outputs_dir
        self.restoration_matrix = None
        self.projection_matrix = None

    def __str__(self):
        return self.name

    def learn_encoder_decoder(self, data):
        """
        :param data: n x d matrix
        :param laten_dim: size of the latent space of the desired encodings
        :return: (d x laten_dim, laten_dim x d) matrices for encoding decoding data into and from the latent space
        """
        raise NotImplementedError

    def encode(self, zero_mean_data):
        return np.dot(zero_mean_data, self.projection_matrix)

    def decode(self, zero_mean_data):
        return np.dot(zero_mean_data, self.restoration_matrix)

    def get_reconstuction_loss(self, zero_mean_data):
        return np.linalg.norm(zero_mean_data - self.decode(self.encode(zero_mean_data)), ord=2) / zero_mean_data.shape[0]

    def get_orthonormality_loss(self, zero_mean_data):
        PM_PMT = np.dot(self.restoration_matrix, self.projection_matrix)
        return np.linalg.norm(np.identity(self.laten_dim) - PM_PMT, ord=2) / self.laten_dim


class AnalyticalPCA(LinearAutoEncoder):
    def __init__(self, laten_dim):
        super(AnalyticalPCA, self).__init__(laten_dim, "AnalyticalPCA")

    def learn_encoder_decoder(self, data):
        """
        Perform PCA by triming the result orthonormal transformation of SVD
        Assumes X is zero centered
        """
        CovMat = np.dot(data.transpose(),data)

        vals, vecs = np.linalg.eigh(CovMat)

        # Take rows corresponding to highest eiegenvalues
        order = np.argsort(vals,)[::-1][:self.laten_dim]

        self.projection_matrix = vecs[order].transpose()
        self.restoration_matrix = vecs[order]


class NumericMinimizationPCA(LinearAutoEncoder):
    def __init__(self, laten_dim, training_dir, optimization_steps=1000, lr=0.001, regularization_factor=1):
        super(NumericMinimizationPCA, self).__init__(laten_dim, f"NumericMinimizationPCA_s[{optimization_steps}]_lr[{lr}]_r[{regularization_factor}]", training_dir)
        self.optimization_steps = optimization_steps
        self.regularization_factor = regularization_factor
        self.lr = lr

    def learn_encoder_decoder(self, data):
        """
        Perform PCA by a straightforward minimization of ||X - XCC^T|| constraint to C's columns being orthonormal vectors
        i.e C^TC = I.
        This minimization problem is equivalent to maximization of the projections variation i.e the variation in
        featues of XC (XC)^T(XC)
        The analytical solution to this problem is a metrix with XX^T eiegenvalues as columns
        see http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/
        Assumes X is zero centered
        """
        X = torch.tensor(data, requires_grad=False)
        C = torch.tensor(np.random.normal(0, 1, (data.shape[1], self.laten_dim)), requires_grad=True)
        optimizer = torch.optim.Adam([C], lr=self.lr)

        losses = [[], []]

        for s in tqdm(range(self.optimization_steps)):
            projected_data = torch.matmul(X, C)
            reconstruct_data = torch.matmul(projected_data, C.t())
            reconstruction_loss = torch.nn.functional.mse_loss(X, reconstruct_data)

            # ensure C columns are orthonormal
            CT_C = torch.matmul(C.t(), C)
            constraint_loss = torch.nn.functional.mse_loss(CT_C, torch.eye(self.laten_dim, dtype=C.dtype))

            loss = reconstruction_loss + self.regularization_factor * constraint_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [reconstruction_loss.item()]
            losses[1] += [constraint_loss.item()]

        # plot training
        if self.outputs_dir:
            plot_training(losses, ["reconstruction_loss", "constrain_loss"], self.outputs_dir, self.name)

        C = C.detach().numpy()

        # Sort PCs in descending order by eiegenvalues
        eiegenvalues = []
        for i in range(self.laten_dim):
            data_projected_to_pc = np.dot(data, C[:, i])
            pc_variation = np.dot(data_projected_to_pc.transpose(), data_projected_to_pc)
            C_norm = np.dot(C[0].transpose(), C[0])
            eiegenvalues += [pc_variation / C_norm]

        order = np.argsort(eiegenvalues)[::-1]
        self.projection_matrix = C[:, order]
        self.restoration_matrix = self.projection_matrix.transpose()


class VanilaAE(LinearAutoEncoder):
    def __init__(self, laten_dim, training_dir, optimization_steps=1000, lr=0.001, batch_size=64):
        super(VanilaAE, self).__init__(laten_dim, f"VanilaAE_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]", training_dir)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.batch_size = batch_size

    def learn_encoder_decoder(self, data):
        """
        SGD minimzation of reconstruction loss ||X - XED|| where E and D are any
        d x laten_dim and laten_dim x d matrices.
        """
        X = torch.tensor(data, requires_grad=False)
        E = torch.tensor(np.random.normal(0, 1, (data.shape[1], self.laten_dim)), requires_grad=True)
        D = torch.tensor(np.random.normal(0, 1, (self.laten_dim, data.shape[1])), requires_grad=True)
        optimizer = torch.optim.Adam([E, D], lr=self.lr)

        losses = [[]]

        for s in tqdm(range(self.optimization_steps)):
            batch_X = X[torch.randint(X.shape[0], (self.batch_size,))]
            projected_data = torch.matmul(batch_X, E)
            reconstruct_data = torch.matmul(projected_data, D)
            loss = torch.nn.functional.mse_loss(batch_X, reconstruct_data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [loss.item()]
        if self.outputs_dir:
            plot_training(losses, ["reconstruction_loss"], self.outputs_dir, self.name)

        self.projection_matrix = E.detach().numpy()
        self.restoration_matrix = D.detach().numpy()


class ALAE(LinearAutoEncoder):
    def __init__(self,laten_dim,  training_dir, z_dim=32, optimization_steps=1000, lr=0.002, batch_size=64):
        super(ALAE, self).__init__(laten_dim, f"ALAE_z_dim[{z_dim}]_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]", training_dir)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.z_dim = z_dim
        self.batch_size = batch_size
        # self.delta =

    def learn_encoder_decoder(self, data):
        """
        SGD minimzation of reconstruction loss ||X - XED|| where E and D are any
        d x laten_dim and laten_dim x d matrices.
        """
        N, d = data.shape
        F = torch.nn.Linear(self.z_dim, self.laten_dim)
        G = torch.nn.Linear(self.laten_dim, d)
        E = torch.nn.Linear(d, self.laten_dim)
        D = torch.nn.Linear(self.laten_dim, 1)
        ED_optimizer = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=self.lr, betas=(0.0, 0.99))
        FG_optimizer = torch.optim.Adam(list(F.parameters()) + list(G.parameters()), lr=self.lr, betas=(0.0, 0.99))
        EG_optimizer = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=self.lr, betas=(0.0, 0.99))

        softplus = torch.nn.functional.softplus
        mse = torch.nn.functional.mse_loss
        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        losses = [[], [], []]

        for s in tqdm(range(self.optimization_steps)):
            # Step I. Update E, and D
            ED_optimizer.zero_grad()
            batch_real_data = X[torch.randint(N, (self.batch_size,))]
            batch_latent_vectors = torch.tensor(np.random.normal(0,1,size=(self.batch_size, self.z_dim)), requires_grad=True, dtype=torch.float32)
            L_adv_ED = softplus(D(E(G(F(batch_latent_vectors))))).mean() + softplus(-D(E(batch_real_data))).mean()
                        # TODO: + R1 gradient regularization as in paper
            L_adv_ED.backward()
            ED_optimizer.step()

            # Step II. Update F, and G
            FG_optimizer.zero_grad()
            batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=True, dtype=torch.float32)
            L_adv_FG = softplus(-D(E(G(F(batch_latent_vectors))))).mean()
            L_adv_FG.backward()
            FG_optimizer.step()

            # Step III. Update E, and G
            EG_optimizer.zero_grad()
            batch_latent_vectors = torch.tensor(np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), requires_grad=True, dtype=torch.float32)
            L_err_EG = mse(F(batch_latent_vectors) , E(G(F(batch_latent_vectors))))
            L_err_EG.backward()
            EG_optimizer.step()

            losses[0] += [L_adv_ED.item()]
            losses[1] += [L_adv_FG.item()]
            losses[2] += [L_err_EG.item()]

        # plot training
        if self.outputs_dir:
            plot_training(losses, ["ED_loss", "FG_loss", 'EG_loss'], self.outputs_dir, self.name)

        self.projection_matrix = E.weight.t().detach().numpy()
        self.restoration_matrix = G.weight.t().detach().numpy()


