import numbers
import numpy as np
import scipy.sparse as sp
import time
import warnings
from math import sqrt
from sklearn.datasets import load_iris

from sklearn._config import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_is_fitted, check_non_negative

from sklearn import svm
from sklearn.metrics import accuracy_score

EPSILON = np.finfo(np.float32).eps ##
# print('eps:',EPSILON)

def norm(x):
    """Dot product-based Euclidean norm implementation.

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm.
    """
    return sqrt(squared_norm(x))

def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError(
            "Array with wrong shape passed to %s. Expected %s, but got %s "
            % (whom, shape, np.shape(A))
        )
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError("Array passed to %s is full of zeros." % whom)

def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix.
    Y : array-like
        Second matrix.
    """
    return np.dot(X.ravel(), Y.ravel())

def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float. 返回的是 参数编号，比如输入frobenius,返回2"""
    allowed_beta_loss = {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError(
            "Invalid beta_loss parameter: got %r instead of one of %r, or a float."
            % (beta_loss, allowed_beta_loss.keys())
        )
    return beta_loss

def _compute_regularization(alpha, alpha_W, alpha_H, l1_ratio, regularization):
    """Compute L1 and L2 regularization coefficients for W and H."""
    if alpha_W != 0 or alpha_H != "same":
        # if alpha_W or alpha_H is not left to its default value we ignore alpha and
        # regularization.
        alpha_H = alpha_W if alpha_H == "same" else alpha_H
        l1_reg_W = alpha_W * l1_ratio
        l1_reg_H = alpha_H * l1_ratio
        l2_reg_W = alpha_W * (1.0 - l1_ratio)
        l2_reg_H = alpha_H * (1.0 - l1_ratio)
    else:
        # TODO remove in 1.2
        l1_reg_W, l2_reg_W, l1_reg_H, l2_reg_H = 0.0, 0.0, 0.0, 0.0
        if regularization in ("both", "transformation"):
            l1_reg_W = alpha * l1_ratio
            l2_reg_W = alpha * (1.0 - l1_ratio)
        if regularization in ("both", "components"):
            l1_reg_H = alpha * l1_ratio
            l2_reg_H = alpha * (1.0 - l1_ratio)

    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H


def _initialize_nmf(X, n_components, init="warn", eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """

    if init == "warn":
        warnings.warn(
            "The 'init' value, when 'init=None' and "
            "n_components is less than n_samples and "
            "n_features, will be changed from 'nndsvd' to "
            "'nndsvda' in 1.1 (renaming of 0.26).",
            FutureWarning,
        )
        init = None

    check_non_negative(X, "NMF initialization")
    n_features, n_samples  = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvd"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype, copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype, copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors 提取列向量的正负部分
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H

def _initializeXlist_nmf(X, n_components, init="warn", eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """

    # Random initialization

    n_features, n_samples= X.shape
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype, copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype, copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors 提取列向量的正负部分
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H

def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(
                axis=1
            )

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)

def _multiplicative_update_w(
    X,
    W,
    H,
    beta_loss,
    l1_reg_W,
    l2_reg_W,
    gamma,
    H_sum=None,
    HHt=None,
    XHt=None,
    update_H=True,
):
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.copy()

        # Denominator 分母
        if HHt is None:
            HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # Numerator
        # if X is sparse, compute WH only where X is non zero 如果X是稀疏的，则仅在X不为零时计算WH
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(WH_safe_X, H.T)

        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    return delta_W, H_sum, HHt, XHt

def _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma):
    """Update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        numerator = safe_sparse_dot(W.T, X)
        denominator = np.linalg.multi_dot([W.T, W, H])

    else:
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH == 0] = EPSILON

        # to avoid division by zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            W_sum = np.sum(W, axis=0)  # shape(n_components, )
            W_sum[W_sum == 0] = 1.0
            denominator = W_sum[:, np.newaxis]

        # beta_loss not in (1, 2)
        else:
            # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
            if sp.issparse(X):
                # memory efficient computation
                # (compute column by column, avoiding the dense matrix WH)
                WtWH = np.empty(H.shape)
                for i in range(X.shape[1]):
                    WHi = np.dot(W, H[:, i])
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WtWH[:, i] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WtWH = np.dot(W.T, WH)
            denominator = WtWH

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_H = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_H **= gamma

    return delta_H

def _beta_divergence(X, W, H, beta, square_root=False):
    """计算X和点（W，H）的β散度。

    Parameters
    ----------
    X : float or array-like of shape (n_samples, n_features)

    W : float or array-like of shape (n_samples, n_components)

    H : float or array-like of shape (n_components, n_features)

    beta : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.

    square_root : bool, default=False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.

    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H).
    """

    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.linalg.multi_dot([W.T, W, H]), H)
            cross_prod = trace_dot((X * H.T), W)
            res = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0
        else:
            res = squared_norm(X - np.dot(W, H)) / 2.0

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res



def _fit_multiplicative_update(
    X,
    W,
    H,
    beta_loss="frobenius",
    max_iter=200,
    tol=1e-4,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
):
    """非负矩阵乘法更新
    目标函数为_beta_dergence（X，WH），并通过W和H的交替最小化来最小化。每个最小化都通过乘法更新来完成。
    Parameters
    ----------
    X : W :  H :

    beta_loss : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}, default='frobenius'
        字符串必须在{“frobenius” frobenius范数（矩阵所有元素的最小和）、“kullback leibler” KL散度、“itakura saito”}中。
        β发散最小化，测量X和点积WH之间的距离。
        请注意，与“frobenius”（或2）和“kullback-leibler”（或1）不同的值会导致拟合明显较慢。
        请注意，对于beta_loss<=0（或“itakura saito”），输入矩阵X不能包含零。
    max_iter : int, default=200 迭代次数
    tol : float, default=1e-4 停止条件的公差。
    l1_reg_W : float, default=0. W的L1正则化参数。
    l1_reg_H : float, default=0. H的L1正则化参数。
    l2_reg_W : float, default=0. W的L2正则化参数。
    l2_reg_H : float, default=0. H的L2正则化参数。
    update_H : bool, default=True  是否对H进行更新。
    verbose : int, default=0  The verbosity level. 冗长程度。
    Returns
    -------
    W : ndarray of shape (n_samples, n_components). 基矩阵
    H : ndarray of shape (n_components, n_features). 投影矩阵
    n_iter : int 矩阵更新的迭代次数
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    #print('bate_loss:', beta_loss,type(beta_loss))

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011] 伽玛最大化最小化（MM）算法
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    # used for the convergence criterion 用于收敛准则
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
            X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma, H_sum, HHt, XHt, update_H
        )
        W *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.0

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(
                X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma
            )
            H *= delta_H

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        # test convergence criterion every 10 iterations 每10次迭代测试收敛准则
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_iter, iter_time - start_time, error)
                )

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
        )

    return W, H, n_iter


################################################################################################################
###### 流形约束，计算L D W
def rbf(dist, t = 1.0):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist/t))

def cal_pairwise_dist(x):

    '''计算pairwise 距离, x是matrix
    (a-b)^2 = a^2 + b^2 - 2*a*b
    '''
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    #返回任意两个点之间距离的平方
    return dist

def cal_rbf_dist(data, n_neighbors = 10, t = 1):
    '''
    拉普拉斯矩阵近邻权重计算
    '''
    dist = cal_pairwise_dist(data) # 计算每个点和其他点的欧式距离

    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)
    # print('rbf_dist:', rbf_dist.tolist())
    G = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1+n_neighbors]
        G[i, index_] = rbf_dist[i, index_]
        G[index_, i] = rbf_dist[index_, i]

    return G

def cal_neighborsWeightMatrix(data, n_neighbors = 10, t = 1):
    '''
    拉普拉斯矩阵近邻权重计算
    '''
    dist = cal_pairwise_dist(data) # 计算每个点和其他点的欧式距离

    print('n_neighbors',n_neighbors)

    dist[dist < 0] = 0
    n = dist.shape[0]
    # print('rbf_dist:', rbf_dist.tolist())
    G = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1+n_neighbors]
        G[i, index_] = 1
        G[index_, i] = 1

    return G


def cos_sim(point1,point2):
    '''
    余弦相似度
    :param point1: 第一个点
    :param point2: 第二个点
    :return:
    '''
    sim = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
    return sim

def cal_cos_sim (data):
    # 数据点，两两计算余弦相似度
    length = data.shape[0]

    cos_s = np.zeros((length, length))

    for i in range(length):
        for j in range(length):
            cos_s[i,j] = cos_sim(data[i],data[j])
    return cos_s

def cal_cos_sim_Martix(data,n_neighbors = 10, t=1.0):
    # dist = cal_pairwise_dist(data)
    cos_s = cal_cos_sim(data)
    n = cos_s.shape[0]
    # cos_dist = rdf2(cos_s,1)
    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(-cos_s[i])[1:1 + n_neighbors]
        W[i, index_] = 1
        W[index_, i] = 1
    return W

def getLaplacianMatrix(X,sy,n_neighbors = 5):
    ''' 调用这个
    欧式距离做近邻矩阵的拉普拉斯矩阵计算，
    '''
    N = X.shape[0]

    # 余弦距离,效果不好
    # G = cal_cos_sim_Martix(X,n_neighbors)
    # 欧式距离 核函数
    # G = cal_rbf_dist(X, n_neighbors)

    # 欧式距离 权重1
    G = cal_neighborsWeightMatrix(X, n_neighbors)

    # print("G",G.tolist()[0])

    # print('G:',G.tolist())

    # 源域标签计算权重
    # wy = np.zeros((len(sy), len(sy)))
    # # print(wy)
    # for i in range(len(sy)):
    #     for j in range(len(sy)):
    #         if sy[i] == sy[j]:
    #             wy[i, j] = 1
    # print('wy.shape', wy.shape)
    # # 标签权重替换欧式距离权重
    # G[0:len(wy), 0:len(wy)] = wy



    D = np.zeros_like(G)
    for i in range(N):
        D[i,i] = np.sum(G[i])
    L = D - G

    return L, D, G


def getDulConsLaplacianMatrix(X,sy,n_neighbors = 5,wtsfactor = 1.5, wtfactor = 0.5):
    ''' 调用这个
    双约束拉普拉斯矩阵
    '''
    N = X.shape[0]
    Ns = len(sy)
    # Ns = 601
    Nt = N - Ns

    Xs = X[:Ns]
    Xt = X[Ns:]

    ## 源域标签计算权重
    ws = np.zeros((Ns, Ns))
    # print(wy)
    # for i in range(Ns):
    #     for j in range(Ns):
    #         if sy[i] == sy[j]:
    #             ws[i, j] = 1


    ## 目标域欧式距离 权重 1 neighbors = 10
    wt = cal_neighborsWeightMatrix(Xt,n_neighbors)

    ## 目标域源域近邻权重

    wtsdist = getSrcTarDisW(Xs,Xt)
    wts = np.zeros(wtsdist.shape)
    for i in range(Nt):
        index_ = np.argsort(wtsdist[i])[:n_neighbors]
        wts[i, index_] = 1

    # print(list(wts))

    wts = wtsfactor * wts
    print('wtsfactor',wtsfactor)

    wt = wtfactor * wt
    print('wtfactor', wtfactor)

    # 源域矩阵和目标域矩阵合并
    w1 = np.hstack((ws,wts.T))
    w2 = np.hstack((wts,wt))
    G = np.vstack((w1,w2))

    print('G.shape',G.shape)

    D = np.zeros_like(G)
    for i in range(N):
        D[i,i] = np.sum(G[i])
    L = D - G

    # print('max LDG',np.max(L),np.max(D),np.max(G))
    return L, D, G


def getSrcTarDisW(Xsrc,Xtar):
    ## 目标域到源域的近邻矩阵
    # print('Xsrc.shape',Xsrc[0].shape)
    # print('Xtar.shape',Xtar[0].shape)
    # print(np.linalg.norm(Xsrc[0]-Xtar[0]))

    Ns = len(Xsrc)
    Nt = len(Xtar)
    Dist = np.zeros((Nt, Ns))

    for ti in range(Nt):
        for si in range(Ns):
            Dist[ti][si] = np.linalg.norm(Xtar[ti]-Xsrc[si])


    return Dist






################################################################################################


class MyMulDNMF:
    def __init__(
            self,
            n_components=None, #降维后的维度
            *,
            init="warn", #  初始化方法{'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}
                         #  `None`: 'nndsvd' if n_components <= min(n_samples, n_features), otherwise random.
                         #  `'random'`: 非负随机矩阵，用公式缩放: sqrt(X.mean() / n_components)
                         #  `'nndsvd'`：非负双奇异值分解（nndsvd）初始化（更适合稀疏性）
                         #  `'nndsvda'`: 用X的平均值填充零的NNDSVD（当不需要稀疏性时更好）
                         #  `'nndsvdar'`: 有用小随机值填充的零的NNDSVD（当不需要稀疏性时，通常更快、更不准确地替代NNDSVDa）
                         #  `'custom'`: 自定义初始的W和H
            solver="mu",  # cd|mu：cd是坐标下降解算器，mu是乘法更新解算器
            beta_loss="frobenius",  # float or {'frobenius', 'kullback-leibler', 'itakura-saito'}, 默认'frobenius'
                                    # β发散最小化，测量X和点积WH之间的距离。请注意，与“frobenius”（或2）和“kullback-leibler”（或1）
                                    # 不同的值会导致拟合明显较慢。请注意，对于beta_loss&lt<=0（或“itakura saito”），
                                    # 输入矩阵X不能包含零。仅在“mu”解算器中使用。
                                    # {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
            tol=1e-4,  # 停止条件的公差。
            max_iter=200,  # 达到超时前的最大迭代次数
            random_state=None,  # 等同于random seed 固定随机数
            alpha="deprecated",  # 正则化参数，已被弃用
            alpha_W=0.0,  # float, default=0.0，乘以“W”的正则化项的常数。将其设置为零（默认值）以在“W”上没有正则化。
            alpha_H="same",  # float or "same", default="same"，乘以“H”的正则化项的常数。将其设置为零，则对“H”没有正则化。
                             # 如果为“same”（默认值），则取与“alpha_W”相同的值。
            l1_ratio=0.0,  # float, default=0.0，正则化混合参数，with 0 <= l1_ratio <= 1. 坐标下降解算器cd中使用的正则化参数*l1_ratio*。
            verbose=0,  # int，default=0，是否冗余
            shuffle=False,  # bool, default=False，如果为true，则随机化CD解算器中的坐标顺序
            regularization="deprecated",
            wtsfactor = 1.5,
            wtfactor = 0.5
    ):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.regularization = regularization
        self.wtsfactor = wtsfactor
        self.wtfactor = wtfactor


    def _check_params(self, X):
        # 检查参数合法性
        # n_components
        # self._n_components = self.n_components
        # if self._n_components is None:
        #     self._n_components = X.shape[1]
        if (
            not isinstance(self._n_components, numbers.Integral)
            or self._n_components <= 0
        ):
            raise ValueError(
                "Number of components must be a positive integer; got "
                f"(n_components={self._n_components!r})"
            )

        # max_iter
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iterations must be a positive "
                f"integer; got (max_iter={self.max_iter!r})"
            )

        # tol
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got "
                f"(tol={self.tol!r})"
            )

        # beta_loss
        self._beta_loss = _beta_loss_to_float(self.beta_loss)
        # print("self._beta_loss",self._beta_loss)

        # alpha and regularization are deprecated in favor of alpha_W and alpha_H
        # TODO clean up in 1.2
        if self.alpha != "deprecated":
            warnings.warn(
                "`alpha` was deprecated in version 1.0 and will be removed "
                "in 1.2. Use `alpha_W` and `alpha_H` instead",
                FutureWarning,
            )
            alpha = self.alpha
        else:
            alpha = 0.0

        if self.regularization != "deprecated":
            warnings.warn(
                "`regularization` was deprecated in version 1.0 and will be "
                "removed in 1.2. Use `alpha_W` and `alpha_H` instead",
                FutureWarning,
            )
            allowed_regularization = ("both", "components", "transformation", None)
            if self.regularization not in allowed_regularization:
                raise ValueError(
                    f"Invalid regularization parameter: got {self.regularization!r} "
                    f"instead of one of {allowed_regularization}"
                )
            regularization = self.regularization
        else:
            regularization = "both"

        (
            self._l1_reg_W,
            self._l1_reg_H,
            self._l2_reg_W,
            self._l2_reg_H,
        ) = _compute_regularization(
            alpha, self.alpha_W, self.alpha_H, self.l1_ratio, regularization
        )

        return self

    def _scale_regularization(self, X):
        n_samples, n_features = X.shape
        if self.alpha_W != 0 or self.alpha_H != "same":
            # if alpha_W or alpha_H is not left to its default value we ignore alpha
            # and regularization, and we scale the regularization terms.
            l1_reg_W = n_features * self._l1_reg_W
            l1_reg_H = n_samples * self._l1_reg_H
            l2_reg_W = n_features * self._l2_reg_W
            l2_reg_H = n_samples * self._l2_reg_H
        else:
            # Otherwise we keep the old behavior with no scaling
            # TODO remove in 1.2
            l1_reg_W = self._l1_reg_W
            l1_reg_H = self._l1_reg_H
            l2_reg_W = self._l2_reg_W
            l2_reg_H = self._l2_reg_H

        return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H

    def _check_w_h(self, X, W, H, update_H):
        # check W and H, or initialize them 检查W和H，或者初始化W和H
        print("self.init",self.init)
        n_samples, n_features = X.shape
        if self.init == "custom" and update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )
        elif not update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got H.dtype = {}.".format(
                        H.dtype
                    )
                )
            # 'mu' solver should not be initialized by zeros
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)
        else:
            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state
            )
        return W, H


    def fit_transform(self, X,lamda, y=None, W=None, H=None, n_neighbors = 10):
        """学习数据X的NMF模型，并返回转换后的数据。
        Parameters
        ----------
        XT: 目标域矩阵
        Xlist : N个源域 list of 矩阵（n_samples，n_features）,[（n_samples，n_features）,（n_samples，n_features）,...]; 特征维度相同
        LambdaList: N个不同源域的权重

        y : 忽略未使用，按惯例为API一致性提供。
        W : 降维后的基矩阵
        H : 投影矩阵，如果update_H=False，则将其用作常数，仅用于求解W。

        Returns
        -------
        W : 基矩阵，也是降维后的数据
        """
        # 原版更新算法
        # W, H, n_iter = self._fit_transform(X, W=W, H=H)

        # 双约束更新算法
        W, H = self.mulS_double_constraint_fit_transform(X, lamda,y=y,W=W, H=H, beta = 2,n_neighbors = n_neighbors)

        # self.reconstruction_err_ = _beta_divergence(
        #     X, W, H, self.beta_loss, square_root=True
        # )

        self.n_components_ = H.shape[0]
        self.components_ = H

        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed


        y : Ignored

        W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like of shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
            If update_H=False, it is used as a constant, to solve for W only.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.


        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.
        """

        #print("W,H：",W,H)

        check_non_negative(X, "NMF (input X)")

        # check parameters 检查参数
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)

        # scale the regularization terms
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._scale_regularization(X)

        # if self.solver == "cd":
        #     W, H, n_iter = _fit_coordinate_descent(
        #         X,
        #         W,
        #         H,
        #         self.tol,
        #         self.max_iter,
        #         l1_reg_W,
        #         l1_reg_H,
        #         l2_reg_W,
        #         l2_reg_H,
        #         update_H=update_H,
        #         verbose=self.verbose,
        #         shuffle=self.shuffle,
        #         random_state=self.random_state,
        #     )
        if self.solver == "mu":
            W, H, n_iter = _fit_multiplicative_update(
                X,
                W,
                H,
                self._beta_loss,
                self.max_iter,
                self.tol,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H=update_H,
                verbose=self.verbose,
            )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence."
                % self.max_iter,
                ConvergenceWarning,
            )

        return W, H, n_iter

    def mulS_double_constraint_fit_transform(self, X, lamda = None,y = None, W=None, H=None, beta_loss="frobenius",
                                             alpha = 1, beta = 10, tol=1e-4,update_H=True,
                                             n_neighbors = 10):
        """ 多源域——双约束 Learn a NMF model for the data X and returns the transformed data.
        Parameters
        ----------
        XT : 源域矩阵 {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed
        Xlist: 源域矩阵列表

        LambdaList： 源域权重列表

        W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.
        H : array-like of shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
            If update_H=False, it is used as a constant, to solve for W only.
        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.
        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.
        n_iter_ : int
            Actual number of iterations.
        """

        # 将beta_loss从字符串转成相应的数字
        beta_loss = _beta_loss_to_float(beta_loss)
        # print('beta_loss',beta_loss)

        # 获取拉普拉斯矩阵相关参数

        wtsf = self.wtsfactor
        wtf = self.wtfactor
        # L,D,G = getLaplacianMatrix(X.T,sy=y,n_neighbors = n_neighbors)
        L, D, G = getDulConsLaplacianMatrix(X.T, sy=y, n_neighbors=n_neighbors, wtsfactor = wtsf, wtfactor= wtf)



        # 检查降维数参数，未定义则默认降到2维
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = 2

        # check parameters 检查参数
        self._check_params(X)

        # if X.min() == 0 and self._beta_loss <= 0:
        #     raise ValueError(
        #         "When beta_loss <= 0 and X contains zeros, "
        #         "the solver may diverge. Please add small values "
        #         "to X, or use a positive beta_loss."
        #     )

        # initialize or check W and H
        #W, H = self._check_w_h(X, W, H, update_H)

        # 对W,H进行初始化
        W, H = _initializeXlist_nmf(X, self.n_components, init=self.init, random_state=self.random_state)


        # scale the regularization terms
        # l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._scale_regularization(X)

        # print('D.shape',D.shape)
        print('初始化后 H.shape', H.shape)

        for n_iter in range(1,self.max_iter):
            # print('n_iter:',n_iter)

            # used for the convergence criterion 用于收敛准则
            temp_error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
            previous_error = temp_error_at_init

            # 计算误差矩阵S和Z
            S,Z = self.getSZ(X,W,H,beta,lamda)


            # 对Wlist里的每一个W进行更新
            delta_W = self.MulS_double_constraint_multiplicationupdata_w(W,H,Z)
            W = W * delta_W

            delta_H = self.MulS_double_constraint_multiplicationupdata_h(W,H,Z,alpha,G,D)
            H *= delta_H

            # print('max(delta_W)', np.max(delta_W),'max(delta_H)', np.max(delta_H),)

            error = _beta_divergence(X, W, H, beta_loss, square_root=True)

            # print('n_iter',n_iter,'收敛误差',error)


            # test convergence criterion every 10 iterations 每10次迭代测试收敛准则
            if tol > 0 and n_iter % 10 == 0:
                error = _beta_divergence(X, W, H, beta_loss, square_root=True)

                if (previous_error - error) / error < tol:
                    break
                previous_error = error



        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence."
                % self.max_iter,
                ConvergenceWarning,
            )

        return W, H

    def getSZ(self,X,W,H,beta,lamda):
        # 初始化A
        A = X - np.dot(W, H)
        # 计算S
        S = np.zeros(A.shape)
        for si in range(S.shape[0]):
            for sj in range(S.shape[1]):
                if A[si, sj] > (beta / (2*lamda)):
                    S[si, sj] = A[si, sj] - (beta / (2*lamda))
                elif A[si, sj] < (-beta / (2*lamda)):
                    S[si, sj] = A[si, sj] + (beta / (2*lamda))
                else:
                    S[si, sj] = 0
        Z = X - S
        return S,Z


    def double_constraint_multiplicationupdata_w(self,W,H,Z):
        # 返回W更新公式的分数部分
        # self._beta_loss == 'frobenius' 2
        # 分子
        numerator = np.dot(Z,H.T)
        # 分母
        denominator = np.dot(np.dot(W,H),H.T)+EPSILON
        deltaW = numerator/denominator

        return deltaW

    def double_constraint_multiplicationupdata_h(self,W,H,Z,alpha,G,D):
        # 返回H更新公式的分数部分
        # self._beta_loss == 'frobenius' 2
        # 分子

        numerator = np.dot(W.T, Z) + alpha*np.dot(H,G)
        # 分母
        denominator = np.dot(np.dot(W.T, W), H) + alpha*np.dot(H,D) + EPSILON
        deltaH = numerator / denominator

        return deltaH

    def MulS_double_constraint_multiplicationupdata_w(self,tmpW,H,tmpZ):
        # 返回W更新公式的分数部分
        # self._beta_loss == 'frobenius' 2
        # 分子
        numerator = np.dot(tmpZ,H.T)
        # 分母
        denominator = np.dot(np.dot(tmpW,H),H.T)+EPSILON
        deltaW = numerator/denominator

        return deltaW

    def MulS_double_constraint_multiplicationupdata_h(self,W,H,Z,alpha,G,D):

        numerator = np.dot(W.T, Z) + alpha * np.dot(H, G)
        # 分母
        denominator = np.dot(np.dot(W.T, W), H) + alpha * np.dot(H, D) + EPSILON
        deltaH = numerator / denominator



        return deltaH

if __name__== '__main__':


    iris = load_iris()
    data = iris.data
    target = iris.target
    print(data.shape)
    print(target.shape)

    N = len(target)
    index = list(np.random.permutation(N))
    # print(index)

    data = data[index, :]
    label = target[index]

    num_train = round(N * 0.9)
    X_train = data[0:num_train, :]
    Y_train = label[0:num_train]
    X_test = data[num_train:N, :, ]
    Y_test = label[num_train:N]
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, Y_train)
    y_test_pred = clf.predict(X_test)
    trainacc = round(accuracy_score(Y_train, clf.predict(X_train)), 4)
    print('train准确率：', trainacc)
    testacc = round(accuracy_score(Y_test, clf.predict(X_test)), 4)
    print('test准确率：', testacc)





    print("复现 NMF 降维后前两行特征")
    X = data.T
    X1 = X+1

    # print(X)
    # print(type(X))
    # can be used for example for dimensionality reduction, source separation or topic extraction
    # 个人认为最重要的参数是n_components、alpha、l1_ratio、solver

    mynmf = MyMulDNMF(n_components=2,  # k value,默认会保留全部特征
              init='nndsvd',  # W H 的初始化方法，包括'random' | 'nndsvd'(默认) |  'nndsvda' | 'nndsvdar' | 'custom'.
              solver='mu',  # 'cd' | 'mu'
              beta_loss='frobenius',  # {'frobenius', 'kullback-leibler', 'itakura-saito'}，一般默认就好
              tol=1e-4,  # 停止迭代的公差
              max_iter=2000,  # 最大迭代次数
              random_state=111,
              l1_ratio=0.,  # 正则化参数
              )

    W = mynmf.fit_transform( X ,lamda=2,y=Y_train)
    H = mynmf.components_

    print(H.shape)
    # print(mynmf.n_iter_)

    drData = H.T
    # N = H.shape[0]
    # label = target
    print(label)
    num_train = round(N * 0.9)
    print(num_train,N)
    X_train = drData[0:num_train, :]
    Y_train = label[0:num_train]
    X_test = drData[num_train:N, :, ]
    Y_test = label[num_train:N]
    print(Y_test)
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(X_train, Y_train)
    y_test_pred = clf.predict(X_test)
    trainacc = round(accuracy_score(Y_train, clf.predict(X_train)), 4)
    print('train准确率：', trainacc)
    testacc = round(accuracy_score(Y_test, clf.predict(X_test)), 4)
    print('test准确率：', testacc)



rbf(1, t = 1.0)

