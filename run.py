import numpy as np

from app.core import main


def _get_dict_from_data(data):
    return {
        'user_id': data[:, 0],
        'item_id': data[:, 1],
        'rating': data[:, 4],
        'confidence': data[:, 5],
    }


def _init_data():
    try:
        train_data = np.load('train.npy')
        valid_data = np.load('valid.npy')
        test_data = np.load('test.npy')
    except:
        train_data = None
        valid_data = None
        test_data = None

    if train_data is None or valid_data is None or test_data is None:
        raw_data = np.load('win_lose_data.npy')
        d = {}
        for i in range(raw_data.shape[0]):
            for j in range(raw_data.shape[1]):
                win = raw_data[i][j][0]
                lose = raw_data[i][j][1]
                if win + lose <= 0:
                    continue
                d[(i, j)] = (win, lose)

        full_data = np.zeros((len(d.keys()), 6))
        for i, (user_id, item_id) in enumerate(d.keys()):
            win, lose = d[(user_id, item_id)]
            full_data[i][0] = user_id
            full_data[i][1] = item_id
            full_data[i][2] = win
            full_data[i][3] = lose
            full_data[i][4] = win / (win + lose)
            full_data[i][5] = win + lose

        mask = np.ones(full_data.shape[0])  # 1: train mask
        mask[np.random.rand(full_data.shape[0]) < 0.02] = 2  # 2: valid mask
        mask[np.random.rand(full_data.shape[0]) < 0.1] = 3  # 3: test mask

        train_data = full_data[mask == 1, :]
        valid_data = full_data[mask == 2, :]
        test_data = full_data[mask == 3, :]

        np.save('train.npy', train_data)
        np.save('valid.npy', valid_data)
        np.save('test.npy', test_data)

    return {
        'train': _get_dict_from_data(train_data),
        'valid': _get_dict_from_data(valid_data),
        'test': _get_dict_from_data(test_data),
    }


if __name__ == '__main__':
    data = _init_data()
    main(data)

    # G = -1 * np.ones((raw_data.shape[0], raw_data.shape[1]))
    # T =
    # for i in range(raw_data.shape[0]):
    #     for j in range(raw_data.shape[1]):
    #         win = raw_data[i, j, 0]
    #         lose = raw_data[i, j, 1]
    #         if win + lose == 0:
    #             continue
    #         if win + lose < 10:
    #             continue
    #         d[win + lose] = d.get(win + lose, 0) + 1
    #
    # # for key in sorted(d.keys()):
    # #     print(key, d[key])
    # # np.save('win_rate.npy', G)
    # # assert False
    #
    # rank = 5
    # # G = np.dot(np.random.rand(1000, rank), np.random.rand(rank, 100))
    # # G[np.random.rand(*G.shape) <= 0.5] = -1
    # print('RANK', rank)
    # test_p = 0.1
    # print(np.amax(G))
    # print(np.amin(G))
    #
    # # mask_ij =  1 if entry (i, j) is training case
    # # mask_ij = -1 if entry (i, j) is test case
    # # mask_ij =  0 otherwise
    # mask = np.ones(G.shape)
    # mask[np.random.rand(*G.shape) <= test_p] = -1
    # mask[G == -1] = 0
    # # G is ground truth
    # print(np.sum(mask == 1))
    # print(np.sum(mask == -1))
    # print(np.sum(mask == 0))
    #
    # # M is problem (training case)
    # M = G.copy()
    # M[mask != 1] = -1
    #
    # # A is answer.
    # # A = _algorithm_average(M, mask)
    # A = _get_A_from_M(M, mask, rank)
    # A = np.maximum(np.minimum(1, A), 0)
    #
    # training_error = G - A
    # training_error[mask != 1] = 0
    # print('TRAINING RMSE',
    #       np.linalg.norm(training_error, 'fro') / math.sqrt(np.sum(mask == 1)))
    #
    # test_error = G - A
    # test_error[mask != -1] = 0
    # print(test_error)
    # print(np.amax(test_error))
    # print(np.amin(test_error))
    # print('TEST RMSE',
    #       np.linalg.norm(test_error, 'fro') / math.sqrt(np.sum(mask == -1)))
