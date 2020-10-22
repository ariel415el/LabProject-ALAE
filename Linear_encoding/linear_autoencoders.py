from autoencoders import *
from utils import plot_training

from sklearn.decomposition import PCA
from scipy.linalg import eigh


class LinearAutoEncoder(EncoderDecoder):
    def __init__(self, data_dim, laten_dim):
        super(LinearAutoEncoder, self).__init__(data_dim, laten_dim)
        self.restoration_matrix = None
        self.projection_matrix = None

    def encode(self, data):
        return np.dot(data, self.projection_matrix)

    def decode(self, data):
        return np.dot(data, self.restoration_matrix)

    def get_reconstuction_loss(self, zero_mean_data):
        return np.linalg.norm(zero_mean_data - self.decode(self.encode(zero_mean_data)), ord=2) / zero_mean_data.shape[0]

    def get_orthonormality_loss(self):
        PM_PMT = np.dot(self.restoration_matrix, self.projection_matrix)
        return np.linalg.norm(np.identity(self.latent_dim) - PM_PMT, ord=2) / self.latent_dim

    def learn_encoder_decoder(self, data, plot_path=None):
        raise NotImplementedError


class AnalyticalPCA(LinearAutoEncoder):
    def __init__(self, data_dim, latent_dim):
        super(AnalyticalPCA, self).__init__(data_dim, latent_dim)
        self.name = "AnalyticalPCA"
        self.train_mean = None

    def learn_encoder_decoder(self, train_samples, plot_path=None):
        """
        Perform PCA by triming the result orthonormal transformation of SVD
        Assumes X is zero centered
        """
        start = time()
        print("\tLearning encoder decoder... ",end="")
        self.train_mean = train_samples.mean(0)
        data = train_samples - self.train_mean

        CovMat = np.dot(data.transpose(), data)

        # vals_np, vecs_np = np.linalg.eigh(CovMat)
        # # Take rows corresponding to highest eiegenvalues
        # order = np.argsort(vals_np,)[::-1][:self.latent_dim]
        # self.projection_matrix = vecs_np[order].transpose()
        # self.restoration_matrix = vecs_np[order]

        vals, vecs = eigh(CovMat, subset_by_index=[self.data_dim - self.latent_dim, self.data_dim - 1])
        self.projection_matrix = vecs
        self.restoration_matrix = vecs.transpose()

        print(f"Finished in {time() - start:.2f} sec")

    def encode(self, data):
        zero_mean_data = data - self.train_mean
        return super(AnalyticalPCA, self).encode(zero_mean_data)

    def decode(self, features):
        data = super(AnalyticalPCA, self).decode(features)
        return data + self.train_mean


class SKLearnPCA(LinearAutoEncoder):
    def __init__(self, data_dim, latent_dim):
        super(SKLearnPCA, self).__init__(data_dim, latent_dim)
        self.pca = PCA(n_components=latent_dim)
        self.name = "SKLearnPCA"

    def learn_encoder_decoder(self, train_samples, plot_path=None):
        self.pca.fit(train_samples)

    def encode(self, data):
        if len(data.shape) == 1:
            return self.pca.transform([data])[0]
        else:
            return self.pca.transform(data)

    def decode(self, features):
        if len(features.shape) == 1:
            return self.pca.inverse_transform([features])[0]
        else:
            return self.pca.inverse_transform(features)

    def get_orthonormality_loss(self):
        return 0


class NumericMinimizationPCA(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim, optimization_steps=1000, lr=0.001, regularization_factor=1):
        super(NumericMinimizationPCA, self).__init__(data_dim, laten_dim)
        self.optimization_steps = optimization_steps
        self.regularization_factor = regularization_factor
        self.lr = lr
        self.train_mean = None
        self.name = f"NumericMinimizationPCA_s[{optimization_steps}]_lr[{lr}]_r[{regularization_factor}]"


    def learn_encoder_decoder(self, train_samples, plot_path=None):
        """
        Perform PCA by a straightforward minimization of ||X - XCC^T|| constraint to C's columns being orthonormal vectors
        i.e C^TC = I.
        This minimization problem is equivalent to maximization of the projections variation i.e the variation in
        featues of XC (XC)^T(XC)
        The analytical solution to this problem is a metrix with XX^T eiegenvalues as columns
        see http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/
        Assumes X is zero centered
        """
        self.train_mean = train_samples.mean(0)
        data = train_samples - self.train_mean

        X = torch.tensor(data, requires_grad=False)
        C = torch.tensor(np.random.normal(0, 1, (data.shape[1], self.latent_dim)), requires_grad=True)
        optimizer = torch.optim.Adam([C], lr=self.lr)

        losses = [[], []]

        for s in range(self.optimization_steps):
            projected_data = torch.matmul(X, C)
            reconstruct_data = torch.matmul(projected_data, C.t())
            reconstruction_loss = torch.nn.functional.mse_loss(X, reconstruct_data)

            # ensure C columns are orthonormal
            CT_C = torch.matmul(C.t(), C)
            constraint_loss = torch.nn.functional.mse_loss(CT_C, torch.eye(self.latent_dim, dtype=C.dtype))

            loss = reconstruction_loss + self.regularization_factor * constraint_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses[0] += [reconstruction_loss.item()]
            losses[1] += [constraint_loss.item()]

        # plot training
        if plot_path:
            plot_training(losses, ["reconstruction_loss", "constrain_loss"], plot_path)

        C = C.detach().numpy()

        # Sort PCs in descending order by eiegenvalues
        eiegenvalues = []
        for i in range(self.latent_dim):
            data_projected_to_pc = np.dot(data, C[:, i])
            pc_variation = np.dot(data_projected_to_pc.transpose(), data_projected_to_pc)
            C_norm = np.dot(C[0].transpose(), C[0])
            eiegenvalues += [pc_variation / C_norm]

        order = np.argsort(eiegenvalues)[::-1]
        self.projection_matrix = C[:, order]
        self.restoration_matrix = self.projection_matrix.transpose()

    def encode(self, data):
        data -= self.train_mean
        return super(NumericMinimizationPCA, self).encode(data)

    def decode(self, features):
        data = super(NumericMinimizationPCA, self).decode(features)
        return data + self.train_mean


class LinearVanilaAE(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim, optimization_steps=1000, lr=0.001, batch_size=64, metric='l2'):
        super(LinearVanilaAE, self).__init__(data_dim, laten_dim)
        self.VanillaVAE = VanilaAE(data_dim, laten_dim, optimization_steps=optimization_steps,
                                   lr=lr, batch_size=batch_size, mode="Linear", metric=metric)

        self.name = "Linear-" + self.VanillaVAE.name
        self.train_mean = None

    def learn_encoder_decoder(self, train_samples, plot_path=None):
        self.train_mean = train_samples.mean(0)
        data = train_samples - self.train_mean

        self.VanillaVAE.learn_encoder_decoder(data, plot_path)

        self.projection_matrix = self.VanillaVAE.E.weight.t().detach().numpy()
        self.restoration_matrix = self.VanillaVAE.D.weight.t().detach().numpy()

    def encode(self, data):
        zero_mean_data = data - self.train_mean
        return super(LinearVanilaAE, self).encode(zero_mean_data)

    def decode(self, features):
        data = super(LinearVanilaAE, self).decode(features)
        return data + self.train_mean


class LinearALAE(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim, z_dim=32, optimization_steps=1000, lr=0.002, batch_size=64):
        super(LinearALAE, self).__init__(data_dim, laten_dim)
        self.ALAE = ALAE(data_dim, laten_dim, z_dim=z_dim, mode="Linear",
                         optimization_steps=optimization_steps, lr=lr, batch_size=batch_size)

        self.name = "Linear-" + self.ALAE.name
        self.train_mean = None

    def learn_encoder_decoder(self, train_samples, plot_path=None):
        self.train_mean = train_samples.mean(0)
        data = train_samples - self.train_mean

        self.ALAE.learn_encoder_decoder(data, plot_path)

        self.projection_matrix = self.ALAE.E.weight.t().detach().numpy()
        self.restoration_matrix = self.ALAE.G.weight.t().detach().numpy()

    def encode(self, data):
        zero_mean_data = data - self.train_mean
        return super(LinearALAE, self).encode(zero_mean_data)

    def decode(self, features):
        data = super(LinearALAE, self).decode(features)
        return data + self.train_mean


class LinearLatentRegressor(LinearAutoEncoder):
    def __init__(self, data_dim, laten_dim, optimization_steps=1000, lr=0.005, batch_size=64, regressor_training='joint'):
        super(LinearLatentRegressor, self).__init__(data_dim, laten_dim)
        self.LatentRegressor = LatentRegressor(data_dim, laten_dim, mode="Linear",
                                               optimization_steps=optimization_steps, lr=lr, batch_size=batch_size,
                                               regressor_training=regressor_training)

        self.name = "Linear-" + self.LatentRegressor.name
        self.train_mean = None

    def learn_encoder_decoder(self, train_samples, plot_path=None):
        self.train_mean = train_samples.mean(0)
        data = train_samples - self.train_mean

        self.LatentRegressor.learn_encoder_decoder(data, plot_path)

        self.projection_matrix = self.LatentRegressor.E.weight.t().detach().numpy()
        self.restoration_matrix = self.LatentRegressor.G.weight.t().detach().numpy()


    def encode(self, data):
        zero_mean_data = data - self.train_mean
        return super(LinearLatentRegressor, self).encode(zero_mean_data)

    def decode(self, features):
        data = super(LinearLatentRegressor, self).decode(features)
        return data + self.train_mean
