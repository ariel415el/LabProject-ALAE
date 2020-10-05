import torch
from  PCA.synthetic_datasets import *

COLORS = ['r','g','b','c','k']


def analytical_PCA(X, d):
    """
    Perform PCA by triming the result orthonormal transformation of SVD
    Assumes X is zero centered
    """
    CovMat = np.dot(X.transpose(),X)

    vals, vecs = np.linalg.eigh(CovMat)

    # Take rows corresponding to highest eiegenvalues
    order = np.argsort(vals,)[::-1][:d]
    projection_matrix = vecs[order].transpose()

    return projection_matrix


def gradient_PCA_minimize_reconstruction(data, d):
    """
    Perform PCA by a straightforward minimization of ||X - XCC^T|| constraint to C's columns being orthonormal vectors
    i.e C^TC = I.
    This minimization problem is equivalent to maximization of the projections variation i.e the variation in
    featues of XC (XC)^T(XC)
    The analytical solution to this problem is a metrix with XX^T eiegenvalues as columns
    see http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/
    Assumes X is zero centered
    """
    STEPS = 10000
    REGULARIZATION_FACTOR = 1
    X = torch.tensor(data, requires_grad=False)
    C = torch.tensor(np.random.normal(0, 1, (data.shape[1], d)), requires_grad=True)
    optimizer = torch.optim.Adam([C], lr=0.001)

    for s in range(STEPS):
        projected_data = torch.matmul(X, C)
        reconstruct_data = torch.matmul(projected_data, C.t())
        target_loss = torch.nn.functional.mse_loss(X, reconstruct_data)

        # ensure C columns are orthonormal
        CT_C = torch.matmul(C.t(), C)
        constraint_loss = torch.nn.functional.mse_loss(CT_C, torch.eye(d, dtype=C.dtype))

        loss = target_loss + REGULARIZATION_FACTOR * constraint_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Step {s}: target_loss: {target_loss}, constraint_loss: {constraint_loss}")

    C = C.detach().numpy()

    # Sort PCs in descending order by eiegenvalues
    eiegenvalues = []
    for i in range(d):
        data_projected_to_pc = np.dot(data, C[:, i])
        pc_variation = np.dot(data_projected_to_pc.transpose(), data_projected_to_pc)
        C_norm = np.dot(C[0].transpose(), C[0])
        eiegenvalues += [ pc_variation / C_norm]

    order = np.argsort(eiegenvalues)[::-1]

    return C[:, order]


def understand_pca_2d():
    d=1
    # Get 2d data
    data, colors = get_multi_gaussian_2d_data()

    # plot data
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title("original_data")
    ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=plt.cm.Spectral, s=0.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    mean = data.mean(0)
    zero_mean_data = data - mean

    # analytic pca
    analytic_projection_matrix = analytical_PCA(zero_mean_data, d)
    projected_points = np.dot(np.dot(data, analytic_projection_matrix), analytic_projection_matrix.transpose())
    projected_points = projected_points - projected_points.mean(0) + mean

    # numeric_pca
    numeric_projection_matrix = gradient_PCA_minimize_reconstruction(zero_mean_data, d)
    numeric_projected_points = np.dot(np.dot(data, numeric_projection_matrix), numeric_projection_matrix.transpose())
    numeric_projected_points = numeric_projected_points - numeric_projected_points.mean(0) + mean

    # plot pcs on data
    pc1 = analytic_projection_matrix[:,0] * 5
    # pc1 = numeric_projection_matrix[:,0] * 5
    ax.plot([mean[0], pc1[0] + mean[0]], [mean[1], pc1[1] + mean[1]], c='red')

    # plot projections
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title("projected-points")
    ax.scatter(projected_points[:,0], projected_points[:,1] , s=0.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_title("numeric_projected-points")
    ax.scatter(numeric_projected_points[:,0], numeric_projected_points[:,1] , s=0.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    plt.show()


def understand_pca_3d(data, colors):
    data_max = np.abs(data).max() * 2
    d = 2

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
    analytic_projection_matrix = analytical_PCA(zero_mean_data, d)

    # numeric_pca
    numeric_projection_matrix = gradient_PCA_minimize_reconstruction(zero_mean_data, d)

    # show reconstruction loss and orthonormality constraint
    for pm, name in zip([analytic_projection_matrix, numeric_projection_matrix], ['analytic', 'numeric']):
        print(f"{name} PCA")
        projected_points = np.dot(zero_mean_data, pm)
        reconstruction_loss = np.linalg.norm(zero_mean_data - np.dot(projected_points, pm.transpose()), ord=2)
        print("\tReconstruction loss: ", reconstruction_loss / data.shape[0]) # / data.shape[0] to match torch mse loss which reduces by mean
        PM_PMT = np.dot(numeric_projection_matrix.transpose(), numeric_projection_matrix)
        constraint_loss = np.linalg.norm(np.identity(d) - PM_PMT, ord=2)
        print("\tOrthonormality loss", constraint_loss)
        # print("\tPM x PM.T", PM_PMT)

    # plot pcs on data
    for pm, ax in zip([analytic_projection_matrix, numeric_projection_matrix], [ax1,ax2]):
        for i in range(d):
            pc = pm[:,i] * data_max * 0.6
            shifted_pc = pc + mean
            ax.plot([mean[0], shifted_pc[0]], [mean[1], shifted_pc[1]], [mean[2], shifted_pc[2]], c=COLORS[i], label=f"pc{i}")
        ax.legend()


    for i, pm in enumerate([analytic_projection_matrix, numeric_projection_matrix]):
        projected_points = np.dot(zero_mean_data, pm)
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