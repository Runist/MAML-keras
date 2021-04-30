# -*- coding: utf-8 -*-
# @File : net.py
# @Author: Runist
# @Time : 2020/7/6 16:52
# @Software: PyCharm
# @Brief: 实现模型分类的网络，MAML与网络结构无关，重点在训练过程

from tensorflow.keras import layers, models, losses
import tensorflow as tf
import numpy as np


class MAML:
    def __init__(self, input_shape, num_classes):
        """
        MAML模型类，需要两个模型，一个是作为真实更新的权重θ，另一个是用来做θ'的更新
        :param input_shape: 模型输入shape
        :param num_classes: 分类数目
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        # 因为不能修改到meta model的权重θ和梯度更新状态，所以在更新θ’时需要用另外一个模型作为载体
        self.meta_model = self.get_maml_model()
        self.inner_writer_step = 0
        self.outer_writer_step = 0

    def get_maml_model(self):
        """
        建立maml模型
        :return: maml model
        """
        model = models.Sequential([
            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu",
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(filters=64, kernel_size=3, padding='same', activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=2, strides=2),

            layers.Flatten(),
            layers.Dense(self.num_classes, activation='softmax'),
        ])

        return model

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None, writer=None):
        """
        MAML一个batch的训练过程
        :param train_data: 训练数据，以task为一个单位
        :param inner_optimizer: support set对应的优化器
        :param inner_step: 内部更新几个step
        :param outer_optimizer: query set对应的优化器，如果对象不存在则不更新梯度
        :param writer: 用于记录tensorboard
        :return: batch query loss
        """
        batch_acc = []
        batch_loss = []

        meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data)
        for support_image, support_label, query_image, query_label in zip(meta_support_image, meta_support_label,
                                                                          meta_query_image, meta_query_label):

            # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
            meta_weights = self.meta_model.get_weights()

            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                    acc = (np.argmax(logits, -1) == query_label).astype(np.int32).mean()

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

                if writer:
                    with writer.as_default():
                        tf.summary.scalar('support_loss', loss, step=self.inner_writer_step)
                        tf.summary.scalar('support_accuracy', acc, step=self.inner_writer_step)
                        self.inner_writer_step += 1

            # 载入support set训练完的模型权重，接下来用来计算query set的loss
            with tf.GradientTape() as tape:
                logits = self.meta_model(query_image, training=True)
                loss = losses.sparse_categorical_crossentropy(query_label, logits)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = (np.argmax(logits, -1) == query_label).astype(np.int32).mean()
                batch_acc.append(acc)

            # 重载最开始的权重，根据上面计算的loss计算梯度和更新方向
            self.meta_model.set_weights(meta_weights)
            if outer_optimizer:
                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

                if writer:
                    with writer.as_default():
                        tf.summary.scalar('query_loss', loss, step=self.outer_writer_step)
                        tf.summary.scalar('query_accuracy', acc, step=self.outer_writer_step)
                        self.outer_writer_step += 1

        return np.array(batch_loss).mean(), np.array(batch_acc).mean()
