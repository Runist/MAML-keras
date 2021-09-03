# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020/7/8 16:54
# @Software: PyCharm
# @Brief: 配置文件
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='Select gpu device.')
parser.add_argument('--train_data_dir', type=str,
                    default="./Omniglot/images_background/",
                    help='The directory containing the train image data.')
parser.add_argument('--val_data_dir', type=str,
                    default="./Omniglot/images_evaluation/",
                    help='The directory containing the validation image data.')
parser.add_argument('--summary_path', type=str,
                    default="./summary",
                    help='The directory of the summary writer.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Number of task per train batch.')
parser.add_argument('--val_batch_size', type=int, default=16,
                    help='Number of task per test batch.')
parser.add_argument('--epochs', type=int, default=100,
                    help='The training epochs.')
parser.add_argument('--inner_lr', type=float, default=0.04,
                    help='The learning rate of of the support set.')
parser.add_argument('--outer_lr', type=float, default=0.001,
                    help='The learning rate of of the query set.')

parser.add_argument('--n_way', type=int, default=10,
                    help='The number of class of every task.')
parser.add_argument('--k_shot', type=int, default=1,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=1,
                    help='The number of query set image for every task.')
parser.add_argument('--input_shape', type=tuple, default=(28, 28, 1),
                    help='The image shape of model input.')

args = parser.parse_args()
