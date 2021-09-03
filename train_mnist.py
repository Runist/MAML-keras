# -*- coding: utf-8 -*-
# @File : train_mnist.py
# @Author: Runist
# @Time : 2021/9/3 9:25
# @Software: PyCharm
# @Brief:
from net import MAML
from tensorflow.keras import datasets, losses, optimizers, metrics
from config import args
import numpy as np
import tensorflow as tf
import os


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    maml = MAML(args.input_shape, 10)
    model = maml.get_maml_model()

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    # 训练teacher网络
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )

    model.fit(x_train, y_train, epochs=3, shuffle=True, batch_size=256 )
    model.evaluate(x_test, y_test)
    model.save_weights("mnist.h5")
