# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

import warnings
from time import time
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from numbers import Integral, Real
from ..neighbors import NearestNeighbors
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils.validation import check_non_negative
from ..utils._param_validation import Interval, StrOptions, Hidden
from ..decomposition import PCA
from ..metrics.pairwise import pairwise_distances, _VALID_METRICS

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
from . import _utils  # type: ignore

# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from . import _barnes_hut_tsne  # type: ignore

MACHINE_EPSILON = np.finfo(np.double).eps

def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.

    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).

    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.

    desired_perplexity : float
        Desired perplexity of the joint probability distributions.

    verbose : int
        Verbosity level.

    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose
    )
    assert np.all(np.isfinite(conditional_P)), "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix(
        (conditional_P.ravel(), distances.indices, distances.indptr),
        shape=(n_samples, n_samples),
    )
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s".format(duration))
    return P

def _kl_divergence_bh(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    skip_num_points=0,
    verbose=False,
    compute_error=True,
    num_threads=1,
):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.

    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.

    P : sparse matrix of shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized. Matrix should be of CSR format.

    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.

    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    angle : float, default=0.5
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    verbose : int, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    num_threads : int, default=1
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        dof=degrees_of_freedom,
        compute_error=compute_error,
        num_threads=num_threads,
    )
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad

def _gradient_descent(
    objective,
    p0,
    it,
    n_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    """Batch gradient descent with momentum and individual gains.

    Parameters
    ----------
    objective : callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.

    p0 : array-like of shape (n_params,)
        Initial parameter vector.

    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).

    n_iter : int
        Maximum number of gradient descent iterations.

    n_iter_check : int, default=1
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.

    momentum : float within (0.0, 1.0), default=0.8
        The momentum generates a weight for previous gradients that decays
        exponentially.

    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.

    min_gain : float, default=0.01
        Minimum individual gain for each parameter.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be aborted.

    verbose : int, default=0
        Verbosity level.

    args : sequence, default=None
        Arguments to pass to objective function.

    kwargs : dict, default=None
        Keyword arguments to pass to objective function.

    Returns
    -------
    p : ndarray of shape (n_params,)
        Optimum parameters.

    error : float
        Optimum.

    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc
            grad_norm = linalg.norm(grad)

            if verbose >= 2:
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                break

    return p, error, i

def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
    r"""Indicate to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : {array-like, sparse matrix} of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.
    """
    n_samples = X.shape[0]
    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=metric)
    if metric == "precomputed":
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t


def _tsne(
    P,
    degrees_of_freedom,
    n_samples,
    X_embedded,
    n_components=2,
    n_iter=1000,
    neighbors=None,
    skip_num_points=0,
    n_iter_check=1,
    min_grad_norm=1e-7,
    learning_rate=100,
    early_exaggeration=12.0,
    exploration_n_iter=250,
    angle=0.5,
    verbose=0,
):
    """Runs t-SNE."""
    # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
    # and the Student's t-distributions Q. The optimization algorithm that
    # we use is batch gradient descent with two stages:
    # * initial optimization with early exaggeration and momentum at 0.5
    # * final optimization with momentum at 0.8
    params = X_embedded.ravel()

    opt_args = {
        "it": 0,
        "n_iter_check": n_iter_check,
        "min_grad_norm": min_grad_norm,
        "learning_rate": learning_rate,
        "verbose": verbose,
        "kwargs": dict(skip_num_points=skip_num_points),
        "args": [P, degrees_of_freedom, n_samples, n_components],
        "n_iter_without_progress": exploration_n_iter,
        "n_iter": exploration_n_iter,
        "momentum": 0.5,
    }
    obj_func = _kl_divergence_bh
    opt_args["kwargs"]["angle"] = angle
    # Repeat verbose argument for _kl_divergence_bh
    opt_args["kwargs"]["verbose"] = verbose
    # Get the number of threads for gradient computation here to
    # avoid recomputing it at each iteration.
    opt_args["kwargs"]["num_threads"] = _openmp_effective_n_threads()

    # Learning schedule (part 1): do 250 iteration with lower momentum but
    # higher learning rate controlled via the early exaggeration parameter
    P *= early_exaggeration
    params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
    if verbose:
        print(
            "[t-SNE] KL divergence after %d iterations with early exaggeration: %f"
            % (it + 1, kl_divergence)
        )

    # Learning schedule (part 2): disable early exaggeration and finish
    # optimization with a higher momentum at 0.8
    P /= early_exaggeration
    remaining = n_iter - exploration_n_iter
    if it < exploration_n_iter or remaining > 0:
        opt_args["n_iter"] = n_iter
        opt_args["it"] = it + 1
        opt_args["momentum"] = 0.8
        opt_args["n_iter_without_progress"] = exploration_n_iter
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

    # Save the final number of iterations
    n_iter_ = it

    if verbose:
        print(
            "[t-SNE] KL divergence after %d iterations: %f"
            % (it + 1, kl_divergence)
        )

    X_embedded = params.reshape(n_samples, n_components)
    # self.kl_divergence_ = kl_divergence

    return X_embedded


# Control the number of exploration iterations with early_exaggeration on
_EXPLORATION_N_ITER = 250

# Control the number of iterations between progress checks
_N_ITER_CHECK = 50

def tsne(
        X,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="pca",
        verbose=0,
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
        square_distances="deprecated",
        comm=None,
    ):

    if learning_rate == "auto":
        # See issue #18018
        learning_rate_ = X.shape[0] / early_exaggeration / 4
        learning_rate_ = np.maximum(learning_rate_, 50)
    else:
        learning_rate_ = learning_rate

    if method != "barnes_hut":
        raise ValueError("'method' must be 'barnes_hut'")

    if method == "barnes_hut" and n_components > 3:
        raise ValueError(
            "'n_components' should be inferior to 4 for the "
            "barnes_hut algorithm as it relies on "
            "quad-tree or oct-tree."
        )
    random_state = check_random_state(random_state)

    n_samples = X.shape[0]

    neighbors_nn = None
    # Compute the number of nearest neighbors to find.
    # LvdM uses 3 * perplexity as the number of neighbors.
    # In the event that we have very small # of points
    # set the neighbors to n - 1.
    n_neighbors = min(n_samples - 1, int(3.0 * perplexity + 1))

    if verbose:
        print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

    # Find the nearest neighbors for every point
    knn = NearestNeighbors(
        algorithm="auto",
        n_jobs=n_jobs,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=metric_params,
    )
    t0 = time()
    knn.fit(X)
    duration = time() - t0
    if verbose:
        print(
            "[t-SNE] Indexed {} samples in {:.3f}s...".format(
                n_samples, duration
            )
        )

    t0 = time()
    distances_nn = knn.kneighbors_graph(mode="distance")
    duration = time() - t0
    if verbose:
        print(
            "[t-SNE] Computed neighbors for {} samples in {:.3f}s...".format(
                n_samples, duration
            )
        )

    # Free the memory used by the ball_tree
    del knn

    # knn return the euclidean distance but we need it squared
    # to be consistent with the 'exact' method. Note that the
    # the method was derived using the euclidean method as in the
    # input space. Not sure of the implication of using a different
    # metric.
    distances_nn.data **= 2

    # compute the joint probability distribution for the input space
    P = _joint_probabilities_nn(distances_nn, perplexity, verbose)

    if isinstance(init, np.ndarray):
        X_embedded = init
    elif init == "pca":
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
        )
        # Always output a numpy array, no matter what is configured globally
        pca.set_output(transform="default")
        X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
        # the default value for random initialization. See issue #18018.
        X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
    elif init == "random":
        # The embedding is initialized with iid samples from Gaussians with
        # standard deviation 1e-4.
        X_embedded = 1e-4 * random_state.standard_normal(
            size=(n_samples, n_components)
        ).astype(np.float32)

    # Degrees of freedom of the Student's t-distribution. The suggestion
    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne(
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded,
        n_components=n_components,
        n_iter=n_iter,
        neighbors=neighbors_nn,
        skip_num_points=0,
        n_iter_check=_N_ITER_CHECK,
        min_grad_norm=min_grad_norm,
        learning_rate=learning_rate_,
        early_exaggeration=early_exaggeration,
        exploration_n_iter=_EXPLORATION_N_ITER,
        angle=angle,
        verbose=verbose,
    )
