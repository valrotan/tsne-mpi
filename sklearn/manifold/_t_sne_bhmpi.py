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
from mpi4py import MPI
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from numbers import Integral, Real
from ..neighbors import NearestNeighbors
from ..base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from ..utils import check_random_state
from ..utils.validation import check_non_negative
from ..utils._param_validation import Interval, StrOptions, Hidden
from ..decomposition import PCA
from ..metrics.pairwise import pairwise_distances, _VALID_METRICS

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
from . import _utils  # type: ignore

# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne_mpi'
from . import _barnes_hut_tsne_mpi  # type: ignore

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
    grad,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    angle=0.5,
    verbose=False,
    compute_error=True,
    comm=None,
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

    verbose : int, default=False
        Verbosity level.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

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

    error = _barnes_hut_tsne_mpi.gradient(
        val_P,
        X_embedded,
        neighbors,
        indptr,
        grad,
        angle,
        n_components,
        verbose,
        rank=comm.Get_rank(),
        size=comm.Get_size(),
        comm=comm,
        dof=degrees_of_freedom,
        compute_error=compute_error,
    )
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    # grad = grad.ravel()
    grad *= c
    
    return error

def _gradient_descent(
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
    comm=None,
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
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    P, degrees_of_freedom, n_samples, n_components = args

    p = p0.copy().ravel()
    pi = np.zeros((n_samples // size * n_components), dtype=np.float32)
    updatei = np.zeros((n_samples // size * n_components), dtype=np.float32)
    update = np.zeros((n_samples * n_components), dtype=np.float32)
    gains = np.ones((n_samples // size * n_components), dtype=np.float32)
    error = np.finfo(float).max

    comm.Scatter(p, pi, root=0)

    if rank == 0:
        best_error = np.finfo(float).max
        best_iter = i = it
        tic = time()
    
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        # sync data
        # comm.Bcast(p, root=0)
        
        gradi = np.zeros((n_samples // size, n_components), dtype=np.float32)

        error = _kl_divergence_bh(p, gradi, *args, comm=comm, **kwargs)

        # collect error and grad
        if kwargs["compute_error"]:
            error = comm.reduce(error, op=MPI.SUM, root=0)
        
        # print(rank, i, gradi.shape)

        # perform update
        gradi = gradi.ravel()
        inc = updatei * gradi < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        gradi *= gains
        updatei = momentum * updatei - learning_rate * gradi
        pi += updatei

        comm.Allgather(pi, p)

        
        if check_convergence:
            grad = None
            if rank == 0:
                grad = np.zeros((size, n_samples//size, n_components), dtype=np.float32)
            comm.Gather(gradi, grad, root=0)

            if rank == 0:
                # sync error
                toc = time()
                duration = toc - tic
                
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
                
                tic = time()
            # elif i - best_iter > n_iter_without_progress:
            #     if verbose >= 2:
            #         print(
            #             "[t-SNE] Iteration %d: did not make any progress "
            #             "during the last %d episodes. Finished."
            #             % (i + 1, n_iter_without_progress)
            #         )
            #     break
            # if grad_norm <= min_grad_norm:
            #     if verbose >= 2:
            #         print(
            #             "[t-SNE] Iteration %d: gradient norm %f. Finished."
            #             % (i + 1, grad_norm)
            #         )
            #     break
        # comm.Barrier()

    return p, error, i


def _tsne(
    P,
    degrees_of_freedom,
    n_samples,
    X_embedded,
    n_components=2,
    n_iter=1000,
    n_iter_check=1,
    min_grad_norm=1e-7,
    learning_rate=100,
    early_exaggeration=12.0,
    exploration_n_iter=250,
    angle=0.5,
    verbose=0,
    comm=None,
):
    """Runs t-SNE."""
    # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
    # and the Student's t-distributions Q. The optimization algorithm that
    # we use is batch gradient descent with two stages:
    # * initial optimization with early exaggeration and momentum at 0.5
    # * final optimization with momentum at 0.8

    rank = comm.Get_rank() if comm else 0

    params = X_embedded.ravel()

    opt_args = {
        "it": 0,
        "n_iter_check": n_iter_check,
        "min_grad_norm": min_grad_norm,
        "learning_rate": learning_rate,
        "verbose": verbose,
        "kwargs": dict(
            angle=angle,
            verbose=verbose,
        ),
        "args": [P, degrees_of_freedom, n_samples, n_components],
        "n_iter_without_progress": exploration_n_iter,
        "n_iter": exploration_n_iter,
        "momentum": 0.5,
    }

    # Learning schedule (part 1): do 250 iteration with lower momentum but
    # higher learning rate controlled via the early exaggeration parameter
    P *= early_exaggeration
    params, kl_divergence, it = _gradient_descent(params, **opt_args, comm=comm)
    if rank == 0 and verbose:
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
        params, kl_divergence, it = _gradient_descent(params, **opt_args, comm=comm)

    # Save the final number of iterations
    # n_iter_ = it

    if rank == 0:
        if verbose:
            print(
                "[t-SNE] KL divergence after %d iterations: %f"
                % (it + 1, kl_divergence)
            )

        X_embedded.data = params.reshape(n_samples, n_components).data


# Control the number of exploration iterations with early_exaggeration on
_EXPLORATION_N_ITER = 250

# Control the number of iterations between progress checks
_N_ITER_CHECK = 50

def tsne(
        X,
        Z,
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

    rank = comm.Get_rank()

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

    if rank == 0:
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
    else:
        P = None
    P = comm.bcast(P, root=0)

    if rank == 0:
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
    else:
        X_embedded = np.empty((n_samples, n_components), dtype=np.float32)
    comm.Bcast(X_embedded, root=0)

    # Degrees of freedom of the Student's t-distribution. The suggestion
    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    _tsne(
        P,
        degrees_of_freedom,
        n_samples,
        X_embedded,
        n_components=n_components,
        n_iter=n_iter,
        n_iter_check=_N_ITER_CHECK,
        min_grad_norm=min_grad_norm,
        learning_rate=learning_rate_,
        early_exaggeration=early_exaggeration,
        exploration_n_iter=_EXPLORATION_N_ITER,
        angle=angle,
        verbose=verbose,
        comm=comm,
    )

    Z.data = X_embedded.data
