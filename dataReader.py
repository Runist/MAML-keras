# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/7/7 10:06
# @Software: PyCharm
# @Brief: 数据读取脚本

import random
import numpy as np
import glob
import cv2 as cv
import tensorflow as tf


class MAMLDataLoader:

    def __init__(self, data_path, batch_size, n_way=5, k_shot=1, q_query=1):
        """
        MAML数据读取器
        :param data_path: 数据路径，此文件夹下需要有分好类的子文件夹
        :param batch_size: 有多少个不同的任务
        :param n_way: 一个任务中分为几类
        :param k_shot: 一个类中有几个图片用于Inner looper的训练
        :param q_query: 一个类中有几个图片用于Outer looper的训练
        """
        self.file_list = [f for f in glob.glob(data_path + "**/character*", recursive=True)]
        self.steps = len(self.file_list) // batch_size

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.meta_batch_size = batch_size

    def __len__(self):
        return self.steps

    def get_one_task_data(self):
        """
        获取一个task，一个task内有n_way个类，每个类有k_shot张用于inner训练，q_query张用于outer训练
        :return: support_data, query_data
        """
        img_dirs = random.sample(self.file_list, self.n_way)
        support_data = []
        query_data = []

        support_image = []
        support_label = []
        query_image = []
        query_label = []

        for label, img_dir in enumerate(img_dirs):
            img_list = [f for f in glob.glob(img_dir + "**/*.png", recursive=True)]
            images = random.sample(img_list, self.k_shot + self.q_query)

            # Read support set
            for img_path in images[:self.k_shot]:
                image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                image = image / 255.
                image = np.expand_dims(image, axis=-1)
                support_data.append((image, label))

            # Read query set
            for img_path in images[self.k_shot:]:
                image = cv.imread(img_path, cv.IMREAD_UNCHANGED)
                image = image / 255.
                image = np.expand_dims(image, axis=-1)
                query_data.append((image, label))

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_image.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_image.append(data[0])
            query_label.append(data[1])

        return np.array(support_image), np.array(support_label), np.array(query_image), np.array(query_label)

    def get_one_batch(self):
        """
        获取一个batch的样本，这里一个batch中是以task为个体
        :return: k_shot_data, q_query_data
        """

        while True:
            batch_support_image = []
            batch_support_label = []
            batch_query_image = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):
                support_image, support_label, query_image, query_label = self.get_one_task_data()
                batch_support_image.append(support_image)
                batch_support_label.append(support_label)
                batch_query_image.append(query_image)
                batch_query_label.append(query_label)

            yield np.array(batch_support_image), np.array(batch_support_label), \
                  np.array(batch_query_image), np.array(batch_query_label)
