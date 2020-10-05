from Linear_encoding.datasets import *
from Linear_encoding.methods import *


def main():
    data, colors = get_digits()
    latent_space = 10
    print(f"Data dimension: {data.shape[1]}")
    print(f"Target dimension: {latent_space}")
    mean = data.mean(0)
    zero_mean_data = data - mean
    methods = [AnalyticalPCA(),
               NumericMinimizationPCA(optimization_steps=20000, regularization_factor=10),
               VanilaAE(optimization_steps=20000),
               ALAE(optimization_steps=20000)]
    for method in methods:
        print(f"method: {method}")
        projection_matrix, restoration_matrix = method.compute_encoder_decoder(zero_mean_data, latent_space)
        projected_points = np.dot(zero_mean_data, projection_matrix)
        reconstructed_points = np.dot(projected_points, restoration_matrix)

        reconstruction_loss = np.linalg.norm(zero_mean_data - reconstructed_points, ord=2)
        print("\tReconstruction loss: ", reconstruction_loss / data.shape[0]) # / data.shape[0] to match torch mse loss which reduces by mean
        PM_PMT = np.dot(restoration_matrix, projection_matrix)
        orthonormality_loss = np.linalg.norm(np.identity(latent_space) - PM_PMT, ord=2)
        print("\tOrthonormality loss", orthonormality_loss)

if __name__ == '__main__':
    main()