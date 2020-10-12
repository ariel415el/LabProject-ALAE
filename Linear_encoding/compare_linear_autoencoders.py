import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from datasets import *
from Linear_encoding.linear_autoencoders import *
from method_evaluation.classifiers import train_mlp_classifier, train_svm, evaluate_nearest_neighbor


class my_logger(object):
    def __init__(self, fname):
        self.file = open(fname, "w")

    def log(self, txt):
        self.file.write(txt + "\n")
        print(txt)


def main():
    # Get data
    train_data, train_labels, test_data, test_labels, dataset_name = get_sklearn_digits()
    # train_data, train_labels, test_data, test_labels, dataset_name = get_mnist(data_dir='data')

    latent_dim = 10
    train_epochs=15
    lr=0.002

    output_dir = os.path.join("Linear_encoding/outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    data_dim = train_data.shape[1]
    logger = my_logger(os.path.join(output_dir, "log.txt"))
    logger.log(f"Data dimension: {data_dim}")
    logger.log(f"Target dimension: {latent_dim}")
    logger.log(f"Num samples: train/test {train_data.shape[0]} / {test_data.shape[0]}")

    # Base line: train on original dataset
    train_accuracy, test_accuracy = train_mlp_classifier((train_data, train_labels), (test_data, test_labels),
                                                         epochs=train_epochs, lr=lr,
                                                         plot_path=os.path.join(os.path.join(output_dir, f"{dataset_name}_train_original_data.png")))
    logger.log(f"\n{dataset_name} original data accuracy train/test {train_accuracy:.2f}/{test_accuracy:.2f}")

    # normalizer data
    train_data = train_data - train_data.mean(0)
    test_data = test_data - test_data.mean(0)

    methods = [AnalyticalPCA(data_dim, latent_dim),
               # NumericMinimizationPCA(latent_dim, output_dir, optimization_steps=1000, regularization_factor=10),
               LinearLatentRegressor(train_data.shape[1], latent_dim, output_dir, optimization_steps=10000,
                               regressor_training="separate"),
               LinearLatentRegressor(train_data.shape[1], latent_dim, output_dir, optimization_steps=10000,
                               regressor_training="joint"),
               LinearVanilaAE(data_dim, latent_dim, output_dir, optimization_steps=10000, metric='l1'),
               LinearVanilaAE(data_dim, latent_dim, output_dir, optimization_steps=10000, metric='l2'),
               LinearALAE(data_dim, latent_dim, output_dir, optimization_steps=10000)]

    for method in methods:
        logger.log(f"{method}:")
        # Learn encoding on train data train on it and test on test encodings
        method.learn_encoder_decoder(train_data)
        logger.log(f"\tReconstrucion loss train/test {method.get_reconstuction_loss(train_data):.4f}/{method.get_reconstuction_loss(test_data):.4f}")
        logger.log(f"\tOrthonormality loss train/test {method.get_orthonormality_loss(train_data):.4f}/{method.get_orthonormality_loss(test_data):.4f}")

        # Evaluate encoodings
        projected_train_data = method.encode(train_data)
        projected_test_data = method.encode(test_data)

        # MLP classifier
        train_accuracy, test_accuracy = train_mlp_classifier((projected_train_data, train_labels), (projected_test_data, test_labels),
                                                             epochs=train_epochs, lr=lr,
                                                             plot_path=os.path.join(os.path.join(output_dir, f"{dataset_name}_train_{method}.png")))
        logger.log(f"\t{dataset_name} classifier accuracy train/test {train_accuracy:.2f}/{test_accuracy:.2f}")

        # SVM classifier
        train_accuracy, test_accuracy = train_svm((projected_train_data, train_labels), (projected_test_data, test_labels))
        logger.log(f"\t{dataset_name} SVM accuracy train/test {train_accuracy:.2f}/{test_accuracy:.2f}")

        # 1NN classifier
        logger.log(f"\t{dataset_name} 1NN accuracy train/test {evaluate_nearest_neighbor(projected_train_data, train_labels):2f}/"
                   f"{evaluate_nearest_neighbor(projected_test_data, test_labels):.2f}")


if __name__ == '__main__':
    main()