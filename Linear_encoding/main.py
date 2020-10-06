import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from Linear_encoding.datasets import *
from Linear_encoding.methods import *
from method_evaluation.train_sklearn_digits import train_sklearn_digits


def endcode_data(data, outputs_dir, latent_space):
    plots_dir = os.path.join(outputs_dir, "Training_plots")
    datasets_dir = os.path.join(outputs_dir, "projected_datasets")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    # save_orignial_data
    print("Saving labels to file")
    # np.savetxt(os.path.join(datasets_dir, "original_data.txt"), data)
    np.savetxt(os.path.join(datasets_dir, "labels.txt"), labels)

    methods = [AnalyticalPCA(),
               # NumericMinimizationPCA(plots_dir, optimization_steps=1000, regularization_factor=10),
               VanilaAE(plots_dir, optimization_steps=1000),
               ALAE(plots_dir, optimization_steps=1000)]

    # normalizer data
    mean = data.mean(0)
    zero_mean_data = data - mean

    for method in methods:
        print(f"method: {method}")
        projection_matrix, restoration_matrix = method.compute_encoder_decoder(zero_mean_data, latent_space)
        projected_points = np.dot(zero_mean_data, projection_matrix)
        reconstructed_points = np.dot(projected_points, restoration_matrix)
        reconstruction_loss = np.linalg.norm(zero_mean_data - reconstructed_points, ord=2)
        print("\tReconstruction loss: ", reconstruction_loss / data.shape[0]) # / data.shape[0] to match torch mse loss which reduces by mean

        # orthogonality loss is the distances of the projection matrix times its transpose from I, (0 for orthonormal matrix)
        PM_PMT = np.dot(restoration_matrix, projection_matrix)
        orthonormality_loss = np.linalg.norm(np.identity(latent_space) - PM_PMT, ord=2)
        print("\tOrthonormality loss", orthonormality_loss)

        np.savetxt(os.path.join(datasets_dir,  f"{str(method)}.txt"), projected_points)


if __name__ == '__main__':
    output_dir = os.path.join("Linear_encoding/outputs")
    data, labels, dataset_name = get_sklearn_digits()
    # data, labels, dataset_name = get_mnist(data_dir='Linear_encoding/data')

    latent_space = 10
    print(f"Data dimension: {data.shape[1]}")
    print(f"Target dimension: {latent_space}")
    print(f"Num samples: {data.shape[0]}")

    endcode_data(data, os.path.join(output_dir, dataset_name), latent_space)

    labels_array = np.loadtxt(os.path.join(os.path.join(output_dir, dataset_name, 'projected_datasets', "labels.txt")))
    accuracy = train_sklearn_digits(data, labels, epochs=30)
    print(f"Data: original_data, accuracy: {accuracy}")
    for fname in os.listdir(os.path.join(output_dir, dataset_name, 'projected_datasets')):
        if fname == "labels.txt":
            continue
        data_array = np.loadtxt(os.path.join(os.path.join(output_dir, dataset_name, 'projected_datasets', fname)))
        accuracy = train_sklearn_digits(data_array, labels_array, epochs=30)
        print(f"Data: {fname}, accuracy: {accuracy}")
