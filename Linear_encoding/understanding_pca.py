import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from Linear_encoding.datasets import *
from Linear_encoding.methods import *
COLORS = ['r','g','b','c','k']


def understand_pca_3d(data, colors):
    data_max = np.abs(data).max() * 2
    latent_dim = 2

    # plot data
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("original_data (analytical pc)")
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=plt.cm.Spectral, s=0.5)
    ax1.set_xlim(-data_max, data_max)
    ax1.set_ylim(-data_max, data_max)
    ax1.set_zlim(-data_max, data_max)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("original_data (numeric pc)")
    ax2.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=plt.cm.Spectral, s=0.5)
    ax2.set_xlim(-data_max, data_max)
    ax2.set_ylim(-data_max, data_max)
    ax2.set_zlim(-data_max, data_max)

    mean = data.mean(0)
    zero_mean_data = data - mean

    # analytic pca
    analytic_pca = AnalyticalPCA(latent_dim)
    analytic_pca.learn_encoder_decoder(zero_mean_data)

    numeric_pca = NumericMinimizationPCA(latent_dim, training_dir=None, optimization_steps=1000)
    numeric_pca.learn_encoder_decoder(zero_mean_data)

    # show reconstruction loss and orthonormality constraint
    for ae in [analytic_pca, numeric_pca]:
        print(f"{ae}")
        projected_points = ae.encode(zero_mean_data)
        print("\tReconstruction loss: ", ae.get_reconstuction_loss(zero_mean_data))
        print("\tOrthonormality loss", ae.get_orthonormality_loss(zero_mean_data))

    # plot pcs on data
    for ae, ax in zip([analytic_pca, numeric_pca], [ax1,ax2]):
        for i in range(latent_dim):
            pc = ae.projection_matrix[:,i] * data_max * 0.6  # In pca the columns of the projected matrix are the Principal componens
            shifted_pc = pc + mean
            ax.plot([mean[0], shifted_pc[0]], [mean[1], shifted_pc[1]], [mean[2], shifted_pc[2]], c=COLORS[i], label=f"pc{i}")
        ax.legend()

    # plot projection into 2d
    for i, ae in enumerate([analytic_pca, numeric_pca]):
        projected_points = ae.encode(zero_mean_data)
        ax = fig.add_subplot(2, 2, 3 + i)
        ax.set_title("projected-points")
        ax.scatter(projected_points[:,0], projected_points[:,1] , c=colors, s=0.5)
        ax.set_xlim(-data_max, data_max)
        ax.set_ylim(-data_max, data_max)

    plt.show()


if __name__ == '__main__':
    # main()
    # understand_pca_2d()
    # data, colors = get_digits()
    # data, colors = get_multi_gaussian_3d_data()
    # data, colors = get_swiss_roll()
    data, colors, data_2d = get_synthetic_embedded_data()
    understand_pca_3d(data, colors)