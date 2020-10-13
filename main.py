import os
from datasets import get_mnist, get_sklearn_digits
from autoencoders import VanilaAE, ALAE, LatentRegressor
from utils import simple_logger
import argparse
from Linear_encoding.linear_autoencoders import LinearVanilaAE ,LinearALAE, LinearLatentRegressor, AnalyticalPCA
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
            LinearVanilaAE(data_dim, latent_dim, optimization_steps=1000, metric='l1'),
            LinearVanilaAE(data_dim, latent_dim, optimization_steps=1000, metric='l2'),
            LinearLatentRegressor(data_dim, latent_dim, optimization_steps=1000, regressor_training="separate"),
            LinearLatentRegressor(data_dim, latent_dim, optimization_steps=1000, regressor_training="joint"),
            LinearALAE(data_dim, latent_dim, optimization_steps=1000)
        ]
    else:
        raise NotImplementedError

    return autoencoders


def get_evaluation_methods(linear_autoencoders, logger=None):
    evaluaion_methods = [
        ReconstructionLossEvaluator(),
        FirstNearestNeighbor(),
        SVMClassification(SVC_C=5),
        MLP_classification(epochs=20, batch_size=128, lr=0.002, lr_decay=0.9, log_interval=None),
    ]

    if linear_autoencoders:
        evaluaion_methods += [OrthonormalityEvaluator()]

    if logger:
        logger.log("," + ",".join([m.name for m in evaluaion_methods]))

    return evaluaion_methods


def main(args):
    # Get data
    data, dataset_name = get_dataset(args.dataset)

    autoencoders = get_autoencoders(data[0].shape[1], args.latent_dim, args.linear)

    output_dir = os.path.join("outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # set logger
    logger = simple_logger(os.path.join(output_dir, "results.csv"))

    evaluation_methods = get_evaluation_methods(args.linear, logger)

    run_analysis(autoencoders, data, evaluation_methods, output_dir, logger)


def run_analysis(autoencoders, data,  evaluation_methods, outputs_dir, logger):
    train_data, train_labels, test_data, test_labels = data

    # normalizer data
    train_data = train_data - train_data.mean(0)
    test_data = test_data - test_data.mean(0)

    for ae in autoencoders:
        logger.log(f"{ae}", end="")
        # Learn encoding on train data train on it and test on test encodings
        ae.set_training_dir(outputs_dir)
        ae.learn_encoder_decoder(train_data)

        projected_train_data = ae.encode(train_data)
        projected_test_data = ae.encode(test_data)
        projected_data = (projected_train_data, projected_test_data)

        for evaluator in evaluation_methods:
            result_str = evaluator.evaluate(ae, data, projected_data,
                                            plot_path=os.path.join(outputs_dir, f"{evaluator}_{ae}.png"))
            logger.log(f",{result_str}", end="")
        logger.log("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='digits')
    parser.add_argument("--linear", action='store_true')
    parser.add_argument("--latent_dim", type=int, default=10)

    args = parser.parse_args()

    main(args)


