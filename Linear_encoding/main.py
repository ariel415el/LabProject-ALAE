import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from Linear_encoding.datasets import *
from Linear_encoding.methods import *
from method_evaluation.train_classifier import train_on_data


class my_logger(object):
    def __init__(self, fname):
        self.file = open(fname, "w")

    def log(self, txt):
        self.file.write(txt + "\n")
        print(txt)


def main():
    train_data, train_labels, test_data, test_labels, dataset_name = get_sklearn_digits()
    # train_data, train_labels, test_data, test_labels, dataset_name = get_mnist(data_dir='Linear_encoding/data')

    latent_dim = 10
    train_epochs=10
    lr=0.001

    output_dir = os.path.join("Linear_encoding/outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = my_logger(os.path.join(output_dir, "log.txt"))
    logger.log(f"Data dimension: {train_data.shape[1]}")
    logger.log(f"Target dimension: {latent_dim}")
    logger.log(f"Num samples: train/test {train_data.shape[0]}/ {test_data.shape[0]}")


    # Base line: train on original dataset
    train_accuracy, test_accuracy = train_on_data((train_data, train_labels), (test_data, test_labels),
                                                 epochs=train_epochs, lr=lr,
                                                 plot_path=os.path.join(os.path.join(output_dir, f"{dataset_name}_train_original_data.png")))
    logger.log(f"\n{dataset_name} original data accuracy train/test {train_accuracy:.2f}/{test_accuracy:.2f}")

    # Learn encoding on train data train on it and test on test encodings
    # normalizer data
    train_data = train_data - train_data.mean(0)
    test_data = test_data - test_data.mean(0)

    methods = [AnalyticalPCA(latent_dim),
               # NumericMinimizationPCA(latent_dim, output_dir, optimization_steps=1000, regularization_factor=10),
               VanilaAE(latent_dim, output_dir, optimization_steps=1000),
               ALAE(latent_dim, output_dir, optimization_steps=1000)]

    for method in methods:
        logger.log(f"{method}:")
        # Learn encodings
        method.learn_encoder_decoder(train_data)
        logger.log(f"\tReconstrucion loss train/test {method.get_reconstuction_loss(train_data):.4f}/{method.get_reconstuction_loss(test_data):.4f}")
        logger.log(f"\tOrthonormality loss train/test {method.get_orthonormality_loss(train_data):.4f}/{method.get_orthonormality_loss(test_data):.4f}")

        projected_train_data = method.encode(train_data)
        projected_test_data = method.encode(test_data)

        train_accuracy, test_accuracy = train_on_data((projected_train_data, train_labels), (projected_test_data, test_labels),
                                                      epochs=train_epochs, lr=lr,
                                                      plot_path=os.path.join(os.path.join(output_dir, f"{dataset_name}_train_{method}.png")))
        logger.log(f"\t{dataset_name} accuracy train/test {train_accuracy:.2f}/{test_accuracy:.2f}")


if __name__ == '__main__':
    main()