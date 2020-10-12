from autoencoders import *
from utils import plot_training


class LinearAutoEncoder(EncoderDecoder):
    def __init__(self, data_dim, laten_dim, outputs_dir):
        super(LinearAutoEncoder, self).__init__(data_dim, laten_dim, outputs_dir)
        self.restoration_matrix = None
        self.projection_matrix = None

    def encode(self, zero_mean_data):
        return np.dot(zero_mean_data, self.projection_matrix)

    def decode(self, zero_mean_data):
        return np.dot(zero_mean_data, self.restoration_matrix)

    def get_reconstuction_loss(self, zero_mean_data):
        return np.linalg.norm(zero_mean_data - self.decode(self.encode(zero_mean_data)), ord=2) / zero_mean_data.shape[0]

    def get_orthonormality_loss(self, zero_mean_data):
        PM_PMT = np.dot(self.restoration_matrix, self.projection_matrix)
        return np.linalg.norm(np.identity(self.latent_dim) - PM_PMT, ord=2) / self.latent_dim

    def learn_encoder_decoder(self, data):
        raise NotImplementedError


class AnalyticalPCA(LinearAutoEncoder):
    def __init__(self, data_dim, latent_dim):
        super(AnalyticalPCA, self).__init__(data_dim, latent_dim, None)
        self.name = "AnalyticalPCA"

    def learn_encoder_decoder(self, data):
        """
        Perform PCA by triming the result orthonormal transformation of SVD
        Assumes X is zero centered
        """
        CovMat = np.dot(data.transpose(),data)

        vals, vecs = np.linalg.eigh(CovMat)

        # Take rows corresponding to highest eiegenvalues
        order = np.argsort(vals,)[::-1][:self.latent_dim]

        self.projection_matrix = vecs[order].transpose()
        self.restoration_matrix = vecs[order]


class NumericMinimizationPCA(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim, training_dir, optimization_steps=1000, lr=0.001, regularization_factor=1):
        super(NumericMinimizationPCA, self).__init__(data_dim, laten_dim, training_dir)
        self.optimization_steps = optimization_steps
        self.regularization_factor = regularization_factor
        self.lr = lr
        self.name = f"NumericMinimizationPCA_s[{optimization_steps}]_lr[{lr}]_r[{regularization_factor}]"

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


class LinearVanilaAE(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim, training_dir, optimization_steps=1000, lr=0.001, batch_size=64, metric='l2'):
        super(LinearVanilaAE, self).__init__(data_dim, laten_dim, None)
        self.VanillaVAE = VanilaAE(data_dim, laten_dim, training_dir, optimization_steps=optimization_steps,
                                   lr=lr, batch_size=batch_size, mode="Linear", metric=metric)

        self.name = "Linear" + self.VanillaVAE.name


    def learn_encoder_decoder(self, data):
        self.VanillaVAE.learn_encoder_decoder(data)

        self.projection_matrix = self.VanillaVAE.E.weight.t().detach().numpy()
        self.restoration_matrix = self.VanillaVAE.D.weight.t().detach().numpy()


class LinearALAE(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim,  training_dir, z_dim=32, optimization_steps=1000, lr=0.002, batch_size=64):
        super(LinearALAE, self).__init__(data_dim, laten_dim, None)
        self.ALAE = ALAE(data_dim, laten_dim,  training_dir, z_dim=z_dim, mode="Linear",
                         optimization_steps=optimization_steps, lr=lr, batch_size=batch_size)

        self.name = "Linear" + self.ALAE.name


    def learn_encoder_decoder(self, data):
        self.ALAE.learn_encoder_decoder(data)

        self.projection_matrix = self.ALAE.E.weight.t().detach().numpy()
        self.restoration_matrix = self.ALAE.G.weight.t().detach().numpy()


