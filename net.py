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
        self.meta_model = self.get_maml_model()

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

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
        """
        MAML一个batch的训练过程
        :param train_data: 训练数据，以task为一个单位
        :param inner_optimizer: support set对应的优化器
        :param inner_step: 内部更新几个step
        :param outer_optimizer: query set对应的优化器，如果对象不存在则不更新梯度
        :return: batch query loss
        """
        batch_acc = []
        batch_loss = []
        task_weights = []

        # 用meta_weights保存一开始的权重，并将其设置为inner step模型的权重
        meta_weights = self.meta_model.get_weights()

        meta_support_image, meta_support_label, meta_query_image, meta_query_label = next(train_data)
        for support_image, support_label in zip(meta_support_image, meta_support_label):

            # 每个task都需要载入最原始的weights进行更新
            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_image, training=True)
                    loss = losses.sparse_categorical_crossentropy(support_label, logits)
                    loss = tf.reduce_mean(loss)

                    acc = tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int32) == support_label, tf.float32)
                    acc = tf.reduce_mean(acc)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            # 每次经过inner loop更新过后的weights都需要保存一次，保证这个weights后面outer loop训练的是同一个task
            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            for i, (query_image, query_label) in enumerate(zip(meta_query_image, meta_query_label)):

                # 载入每个task weights进行前向传播
                self.meta_model.set_weights(task_weights[i])

                logits = self.meta_model(query_image, training=True)
                loss = losses.sparse_categorical_crossentropy(query_label, logits)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss)

        # 无论是否更新，都需要载入最开始的权重进行更新，防止val阶段改变了原本的权重
        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss, mean_acc
