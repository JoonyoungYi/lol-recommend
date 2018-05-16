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
                if win + lose < 10:
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
