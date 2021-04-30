# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/7/8 16:54
# @Software: PyCharm
# @Brief: 配置文件

train_data_path = "./Omniglot/images_background/"
test_data_path = "./Omniglot/images_evaluation/"
summary_path = "./summary"

batch_size = 4
val_batch_size = 20
epochs = 1000

inner_lr = 0.01
outer_lr = 0.001

n_way = 5
k_shot = 1
q_query = 1

input_shape = (28, 28, 1)
