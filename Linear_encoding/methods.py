import numpy as np
import torch
import os
import matplotlib.pyplot as plt
COLORS =['r', 'g', 'b']


class LinearAutoEncoder(object):
    def __init__(self, name, outputs_dir=None):
        self.name = name
        self.outputs_dir = outputs_dir

    def __str__(self):
        return self.name

    def compute_encoder_decoder(self, data, laten_dim):
        """
        :param data: n x d matrix
        :param laten_dim: size of the latent space of the desired encodings
        :return: (d x laten_dim, laten_dim x d) matrices for encoding decoding data into and from the latent space
        """
        raise NotImplementedError


def plot_training(losses, names, plots_dir, plot_name):
    fig = plt.figure(figsize=(10, 6))
    for i, (loss_list, name) in enumerate(zip(losses, names)):
        ax = fig.add_subplot(1, len(losses), 1 + i)
        ax.set_title(name)
        ax.plot(np.arange(len(loss_list)), loss_list)
        ax.legend()
    plt.savefig(os.path.join(plots_dir, plot_name + ".png"))


class AnalyticalPCA(LinearAutoEncoder):
    def __init__(self):
        super(AnalyticalPCA, self).__init__("AnalyticalPCA")

    def compute_encoder_decoder(self, data, laten_dim):
        """
        Perform PCA by triming the result orthonormal transformation of SVD
        Assumes X is zero centered
        """
        CovMat = np.dot(data.transpose(),data)

        vals, vecs = np.linalg.eigh(CovMat)

        # Take rows corresponding to highest eiegenvalues
        order = np.argsort(vals,)[::-1][:laten_dim]
        projection_matrix = vecs[order].transpose()
        restoration_matrix = vecs[order]
        return projection_matrix, restoration_matrix


class NumericMinimizationPCA(LinearAutoEncoder):
    def __init__(self, training_dir, optimization_steps=1000, lr=0.001, regularization_factor=1):
        super(NumericMinimizationPCA, self).__init__(f"NumericMinimizationPCA_s[{optimization_steps}]_lr[{lr}]_r[{regularization_factor}]", training_dir)
        self.optimization_steps = optimization_steps
        self.regularization_factor = regularization_factor
        self.lr = lr

    def compute_encoder_decoder(self, data, laten_dim):
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
        C = torch.tensor(np.random.normal(0, 1, (data.shape[1], laten_dim)), requires_grad=True)
        optimizer = torch.optim.Adam([C], lr=self.lr)

        losses = [[], []]

        for s in range(self.optimization_steps):
            projected_data = torch.matmul(X, C)
            reconstruct_data = torch.matmul(projected_data, C.t())
            reconstruction_loss = torch.nn.functional.mse_loss(X, reconstruct_data)

            # ensure C columns are orthonormal
            CT_C = torch.matmul(C.t(), C)
            constraint_loss = torch.nn.functional.mse_loss(CT_C, torch.eye(laten_dim, dtype=C.dtype))

            loss = reconstruction_loss + self.regularization_factor * constraint_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [reconstruction_loss.item()]
            losses[1] += [constraint_loss.item()]

        # plot training
        plot_training(losses, ["reconstruction_loss", "constrain_loss"], self.outputs_dir, self.name)

        C = C.detach().numpy()

        # Sort PCs in descending order by eiegenvalues
        eiegenvalues = []
        for i in range(laten_dim):
            data_projected_to_pc = np.dot(data, C[:, i])
            pc_variation = np.dot(data_projected_to_pc.transpose(), data_projected_to_pc)
            C_norm = np.dot(C[0].transpose(), C[0])
            eiegenvalues += [pc_variation / C_norm]

        order = np.argsort(eiegenvalues)[::-1]
        projection_matrix = C[:, order]

        return projection_matrix, projection_matrix.transpose()


class VanilaAE(LinearAutoEncoder):
    def __init__(self, training_dir, optimization_steps=1000, lr=0.001):
        super(VanilaAE, self).__init__(f"VanilaAE_s[{optimization_steps}]_lr[{lr}]", training_dir)
        self.optimization_steps = optimization_steps
        self.lr = lr

    def compute_encoder_decoder(self, data, laten_dim):
        """
        SGD minimzation of reconstruction loss ||X - XED|| where E and D are any
        d x laten_dim and laten_dim x d matrices.
        """
        X = torch.tensor(data, requires_grad=False)
        E = torch.tensor(np.random.normal(0, 1, (data.shape[1], laten_dim)), requires_grad=True)
        D = torch.tensor(np.random.normal(0, 1, (laten_dim, data.shape[1])), requires_grad=True)
        optimizer = torch.optim.Adam([E, D], lr=self.lr)

        losses = [[]]

        for s in range(self.optimization_steps):
            projected_data = torch.matmul(X, E)
            reconstruct_data = torch.matmul(projected_data, D)
            loss = torch.nn.functional.mse_loss(X, reconstruct_data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [loss.item()]

        plot_training(losses, ["reconstruction_loss"], self.outputs_dir, self.name)

        return E.detach().numpy(), D.detach().numpy()


class ALAE(LinearAutoEncoder):
    def __init__(self, training_dir, z_dim=32, optimization_steps=1000, lr=0.002, batch_size=64):
        super(ALAE, self).__init__(f"ALAE_z_dim[{z_dim}]_s[{optimization_steps}]_lr[{lr}]_b[{batch_size}]", training_dir)
        self.optimization_steps = optimization_steps
        self.lr = lr
        self.z_dim = z_dim
        self.batch_size = batch_size
        # self.delta =

    def compute_encoder_decoder(self, data, laten_dim):
        """
        SGD minimzation of reconstruction loss ||X - XED|| where E and D are any
        d x laten_dim and laten_dim x d matrices.
        """
        N, d = data.shape
        F = torch.nn.Linear(self.z_dim, laten_dim)
        G = torch.nn.Linear(laten_dim, d)
        E = torch.nn.Linear(d, laten_dim)
        D = torch.nn.Linear(laten_dim, 1)
        ED_optimizer = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=self.lr, betas=(0.0, 0.99))
        FG_optimizer = torch.optim.Adam(list(F.parameters()) + list(G.parameters()), lr=self.lr, betas=(0.0, 0.99))
        EG_optimizer = torch.optim.Adam(list(E.parameters()) + list(G.parameters()), lr=self.lr, betas=(0.0, 0.99))

        softplus = torch.nn.functional.softplus
        mse = torch.nn.functional.mse_loss
        X = torch.tensor(data, requires_grad=False, dtype=torch.float32)

        losses = [[], [], []]

        for s in range(self.optimization_steps):
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
        plot_training(losses, ["ED_loss", "FG_loss", 'EG_loss'], self.outputs_dir, self.name)

        encoder = E.weight.t().detach().numpy()
        decoder = G.weight.t().detach().numpy()

        return encoder, decoder
