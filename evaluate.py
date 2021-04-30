# -*- coding: utf-8 -*-
# @File : evaluate.py
# @Author: Runist
# @Time : 2021/4/26 15:42
# @Software: PyCharm
# @Brief: 测试脚本

from tensorflow.keras import optimizers, utils, metrics
import tensorflow as tf
import numpy as np

from dataReader import MAMLDataLoader
from net import MAML
import config as cfg
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    val_data = MAMLDataLoader(cfg.test_data_path, cfg.val_batch_size)

    random_model = MAML(cfg.input_shape, cfg.n_way)
    maml = MAML(cfg.input_shape, cfg.n_way)

    # 对比测试
    optimizer = optimizers.Adam(cfg.inner_lr)
    val_loss, val_acc = random_model.train_on_batch(val_data.get_one_batch(), inner_optimizer=optimizer, inner_step=3)
    print("Model with random initialize weight train for 3 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    optimizer = optimizers.Adam(cfg.inner_lr)
    val_loss, val_acc = random_model.train_on_batch(val_data.get_one_batch(), inner_optimizer=optimizer, inner_step=5)
    print("Model with random initialize weight train for 5 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    optimizer = optimizers.Adam(cfg.inner_lr)
    val_loss, val_acc = random_model.train_on_batch(val_data.get_one_batch(), inner_optimizer=optimizer, inner_step=10)
    print("Model with random initialize weight train for 10 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    maml.meta_model.load_weights("maml.h5")
    optimizer = optimizers.Adam(cfg.inner_lr)
    val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(), inner_optimizer=optimizer, inner_step=3)
    print("Model with maml weight train for 3 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    maml.meta_model.load_weights("maml.h5")
    optimizer = optimizers.Adam(cfg.inner_lr)
    val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(), inner_optimizer=optimizer, inner_step=5)
    print("Model with maml weight train for 5 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))

    maml.meta_model.load_weights("maml.h5")
    optimizer = optimizers.Adam(cfg.inner_lr)
    val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(), inner_optimizer=optimizer, inner_step=10)
    print("Model with maml weight train for 10 step, val loss: {:.4f}, accuracy: {:.4f}.".format(val_loss, val_acc))
