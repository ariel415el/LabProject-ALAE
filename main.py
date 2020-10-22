import os
from datasets import get_mnist, get_sklearn_digits
from autoencoders import VanilaAE, ALAE, LatentRegressor
from utils import simple_logger, plot_tsne, plot_latent_interpolation
import argparse
from Linear_encoding.linear_autoencoders import LinearVanilaAE ,LinearALAE, LinearLatentRegressor, AnalyticalPCA,\
    IdentityAutoEncoder, NumericMinimizationPCA, SKLearnPCA
from method_evaluation.evaluators import *


def get_dataset(dataset_name):
    if dataset_name.lower() == 'digits':
        return get_sklearn_digits()
    elif dataset_name.lower() == "mnist":
        return get_mnist(data_dir='data')


def get_autoencoders(data_dim, latent_dim, linear_autoencoders):
    if linear_autoencoders:
        autoencoders = [
            AnalyticalPCA(data_dim, latent_dim),
            # NumericMinimizationPCA(data_dim, latent_dim, optimization_steps=500),
            # SKLearnPCA(data_dim, latent_dim),
            # LinearVanilaAE(data_dim, latent_dim, optimization_steps=1000, metric='l1'),
            LinearVanilaAE(data_dim, latent_dim, optimization_steps=1000, metric='l2'),
            # LinearLatentRegressor(data_dim, latent_dim, optimization_steps=10000, lr=0.01, regressor_training="separate"),
            # LinearLatentRegressor(data_dim, latent_dim, optimization_steps=10000, lr=0.01, regressor_training="joint"),
            LinearALAE(data_dim, latent_dim, optimization_steps=5000,lr=0.00001, batch_size=128),
            # IdentityAutoEncoder(data_dim, None)
        ]
    else:
        raise NotImplementedError

    return autoencoders


def get_evaluation_methods(linear_autoencoders, logger=None):
    evaluaion_methods = [
        ReconstructionLossEvaluator(),
        # FirstNearestNeighbor(),
        # SVMClassification(SVC_C=5),
        # MLP_classification(epochs=20, batch_size=128, lr=0.002, lr_decay=0.999, log_interval=None),
    ]

    if linear_autoencoders:
        evaluaion_methods += [OrthonormalityEvaluator()]

    if logger:
        logger.log("," + ",".join([m.name for m in evaluaion_methods]))

    return evaluaion_methods


def main(args):
    # Get data
    data, dataset_name = get_dataset(args.dataset)
    dataset_name = f"{dataset_name}-{args.latent_dim}"

    autoencoders = get_autoencoders(data[0].shape[1], args.latent_dim, args.linear)

    output_dir = os.path.join("outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # set logger
    logger = simple_logger(os.path.join(output_dir, "results.csv"))

    evaluation_methods = get_evaluation_methods(args.linear, logger)

    train_data, train_labels, test_data, test_labels = data
    print(f"run_analysis on {len(train_data)} train and {len(test_data)} test samples")

    for ae in autoencoders:
        logger.log(f"{ae}", end="")
        print(ae)

        # Learn encoding on train data train on it and test on test encodings
        ae.learn_encoder_decoder(train_data, os.path.join(output_dir,"Training-autoencoder", f"Learning-{ae}.png"))

        start = time()
        print("\tProjecting Data... ", end="")
        projected_train_data = ae.encode(train_data)
        projected_test_data = ae.encode(test_data)
        print(f"Finished in {time() - start:.2f} sec")

        if args.plot_latent_interpolation:
            start = time()
            print("\tVisualizing latent interpolation... ", end="")
            plot_latent_interpolation(ae, train_data, plot_path=os.path.join(output_dir, "Latent-interpollation", f"{ae}-Train.png"))
            plot_latent_interpolation(ae, test_data, plot_path=os.path.join(output_dir, "Latent-interpollation", f"{ae}-Test.png"))
            print(f"Finished in {time() - start:.2f} sec")

        if args.plot_tsne:
            # Run T-SNE
            start = time()
            print("\tRunning T-SNE... ", end="")
            plot_tsne(projected_train_data, train_labels, os.path.join(output_dir, "T-SNE", f"{ae}-Train.png"))
            plot_tsne(projected_test_data, test_labels, os.path.join(output_dir, "T-SNE", f"{ae}-Test.png"))
            print(f"Finished in {time() - start:.2f} sec")

        projected_data = (projected_train_data, projected_test_data)
        for evaluator in evaluation_methods:
            result_str = evaluator.evaluate(ae, data, projected_data,
                                            plot_path=os.path.join(output_dir, "Evaluation", f"{evaluator}_{ae}.png"))
            logger.log(f",{result_str}", end="")
        logger.log("")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='digits')
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--plot_tsne", action='store_true', default=False)
    parser.add_argument("--plot_latent_interpolation", action='store_true', default=False)

    args = parser.parse_args()

    main(args)


