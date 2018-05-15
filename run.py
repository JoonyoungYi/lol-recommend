import random
import math

import numpy as np
from scipy.linalg import orth

MAX_ITER = 10000
EARLY_STOP_ERR = 1e-3
EARLY_STOP_ERR_DELTA = 1e-3


def _make_lr_matrix(m, n, k):
    L = np.random.randn(m, k)
    R = np.random.randn(k, n)
    return np.dot(L, R)


def _get_masked_matrix(M, omega):
    M_max, M_min = np.max(M), np.min(M)
    M_ = M.copy()
    M_[(1 - omega).astype(np.int16)] = M_max * M_max
    return M_


def _get_V_from_U(M, U, mask):
    column = M.shape[1]
    rank = U.shape[1]
    V = np.empty((rank, column), dtype=M.dtype)

    for j in range(0, column):
        U_ = U.copy()
        U_[mask[:, j] != 1, :] = 0
        V[:, j] = np.linalg.lstsq(U_, M[:, j], rcond=None)[0]
    return V


def _get_training_err(M, U, V, mask):
    error_matrix = M - np.dot(U, V)
    error_matrix[mask != 1] = 0
    return np.linalg.norm(error_matrix, 'fro') / math.sqrt(np.sum(mask == 1))


def _init_U(M, rank):
    M[M == -1] = 0
    U, S, V = np.linalg.svd(M, full_matrices=False)
    U_hat = U[:, :rank]
    # clip_threshold = 2 * mu * math.sqrt(k / max(M.shape))
    # U_hat[U_hat > clip_threshold] = 0
    U_hat = orth(U_hat)
    return U_hat


def _get_A_from_M(M, mask, rank):
    U = _init_U(M[:, :], rank)
    V = None
    prev_err = float("inf")
    for t in range(MAX_ITER):
        print('>> ITER', t)

        V = _get_V_from_U(M, U, mask)
        print('   - V:', _get_training_err(M, U, V, mask))
        U = _get_V_from_U(M.T, V.T, mask.T).T
        err = _get_training_err(M, U, V, mask)
        print('   - U:', err)

        if err < EARLY_STOP_ERR:
            break
        if prev_err - EARLY_STOP_ERR_DELTA <= err <= prev_err + EARLY_STOP_ERR_DELTA:
            break

        prev_err = err

    assert V is not None
    return np.dot(U, V)


if __name__ == '__main__':
    try:
        G = np.load('win_rate.npy')
    except:
        G = None

    if G is None:
        raw_data = np.load('win_lose_data.npy')
        print(raw_data.shape)

        G = -1 * np.ones((raw_data.shape[0], raw_data.shape[1]))
        for i in range(raw_data.shape[0]):
            for j in range(raw_data.shape[1]):
                win = raw_data[i, j, 0]
                lose = raw_data[i, j, 1]
                if win + lose == 0:
                    continue
                G[i, j] = win / (win + lose)

        np.save('win_rate.npy', G)

    rank = 3
    print('RANK', rank)
    test_p = 0.1

    # mask_ij =  1 if entry (i, j) is training case
    # mask_ij = -1 if entry (i, j) is test case
    # mask_ij =  0 otherwise
    mask = np.ones(G.shape)
    mask[np.random.rand(*G.shape) <= test_p] = -1
    mask[G == -1] = 0
    # G is ground truth

    # M is problem (training case)
    M = G.copy()
    M[mask != 1] = -1

    # A is answer.
    A = _get_A_from_M(M, mask, rank)
    A = np.maximum(np.minimum(np.ones(A.shape), A), np.zeros(A.shape))

    training_error = G - A
    training_error[mask != 1] = 0
    print('TRAINING RMSE',
          np.linalg.norm(training_error, 'fro') / math.sqrt(np.sum(mask == 1)))

    test_error = G - A
    test_error[mask != -1] = 0
    print('TEST RMSE',
          np.linalg.norm(test_error, 'fro') / math.sqrt(np.sum(mask == -1)))
