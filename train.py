# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2021/4/23 17:30
# @Software: PyCharm
# @Brief: 训练脚本

from tensorflow.keras import optimizers, utils
import tensorflow as tf
import numpy as np

from dataReader import MAMLDataLoader
from net import MAML
from config import args
import shutil
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_data = MAMLDataLoader(args.train_data_dir, args.batch_size)
    val_data = MAMLDataLoader(args.val_data_dir, args.val_batch_size)

    inner_optimizer = optimizers.Adam(args.inner_lr)
    outer_optimizer = optimizers.Adam(args.outer_lr)

    maml = MAML(args.input_shape, args.n_way)
    # 验证次数可以少一些，不需要每次都更新这么多
    val_data.steps = 10

    for e in range(args.epochs):

        train_progbar = utils.Progbar(train_data.steps)
        val_progbar = utils.Progbar(val_data.steps)
        print('\nEpoch {}/{}'.format(e+1, args.epochs))

        train_meta_loss = []
        train_meta_acc = []
        val_meta_loss = []
        val_meta_acc = []

        for i in range(train_data.steps):
            batch_train_loss, acc = maml.train_on_batch(train_data.get_one_batch(),
                                                        inner_optimizer,
                                                        inner_step=1,
                                                        outer_optimizer=outer_optimizer)

            train_meta_loss.append(batch_train_loss)
            train_meta_acc.append(acc)
            train_progbar.update(i+1, [('loss', np.mean(train_meta_loss)),
                                       ('accuracy', np.mean(train_meta_acc))])

        for i in range(val_data.steps):
            batch_val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(), inner_optimizer, inner_step=3)

            val_meta_loss.append(batch_val_loss)
            val_meta_acc.append(val_acc)
            val_progbar.update(i+1, [('val_loss', np.mean(val_meta_loss)),
                                     ('val_accuracy', np.mean(val_meta_acc))])

        maml.meta_model.save_weights("maml.h5")
