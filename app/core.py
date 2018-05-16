import os
import time

import tensorflow as tf
import numpy as np
import pandas as pd

from .configs import *
from .models import init_models

EPOCH_NUMBER = 10000
EARLY_STOP = True
EARLY_STOP_MAX_ITER = 40


def _train(session, saver, models, train_data, valid_data):
    model_file_path = _init_model_file_path()
    prev_valid_rmse = float("Inf")
    early_stop_iters = 0

    for epoch in range(EPOCH_NUMBER):
        if models['train_op']:
            _, train_rmse = session.run(
                [models['train_op'], models['rmse']],
                feed_dict={
                    models['u']: train_data['user_id'],
                    models['i']: train_data['item_id'],
                    models['r']: train_data['rating'],
                    models['c']: train_data['confidence'],
                })
        else:
            train_rmse = float("NaN")

        _, valid_rmse, mu = session.run(
            [models['loss'], models['rmse'], models['mu']],
            feed_dict={
                models['u']: valid_data['user_id'],
                models['i']: valid_data['item_id'],
                models['r']: valid_data['rating'],
                models['c']: valid_data['confidence'],
            })
        # print(mu)

        # if epoch % 10 == 0:
        print('>> EPOCH:', "{:3d}".format(epoch), "{:3f}, {:3f}".format(
            train_rmse, valid_rmse))

        if EARLY_STOP:
            early_stop_iters += 1
            if valid_rmse < prev_valid_rmse:
                prev_valid_rmse = valid_rmse
                early_stop_iters = 0
                saver.save(session, model_file_path)
            elif early_stop_iters >= EARLY_STOP_MAX_ITER:
                print("Early stopping ({} vs. {})...".format(
                    prev_valid_rmse, valid_rmse))
                break
        else:
            saver.save(session, model_file_path)

    return model_file_path


def _test(session, models, valid_data, test_data):
    valid_rmse = session.run(
        [models['rmse']],
        feed_dict={
            models['u']: valid_data['user_id'],
            models['i']: valid_data['item_id'],
            models['r']: valid_data['rating'],
            models['c']: valid_data['confidence'],
        })

    test_rmse = session.run(
        [models['rmse']],
        feed_dict={
            models['u']: test_data['user_id'],
            models['i']: test_data['item_id'],
            models['r']: test_data['rating'],
            models['c']: test_data['confidence'],
        })
    print("Final valid RMSE: {}, test RMSE: {}".format(valid_rmse, test_rmse))
    return valid_rmse, test_rmse


def _init_model_file_path():
    folder_path = 'logs/{}'.format(int(time.time() * 1000))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return os.path.join(folder_path, 'model.ckpt')


def main(data):
    K = 1
    print("rank", K)
    lambda_value = 0.1
    N, M = 560200, 140
    models = init_models(N, M, K, lambda_value)

    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model_file_path = _train(session, saver, models, data['train'],
                                 data['valid'])

        print('Loading best checkpointed model')
        saver.restore(session, model_file_path)
        valid_rmse, test_rmse = _test(session, models, data['valid'],
                                      data['test'])
