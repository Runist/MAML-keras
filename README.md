# Keras  - MAML

## Part 1. Introduction

As we all know, deep learning need vast data. If you don't have this condition, you can use pre-training weights. Most of data can be fitted be pre-training weights,  but there all still some data that can't converge to the global lowest point. So it is exist one weights that can let all task get best result?

Yes, this is "Model-Agnostic Meta-Learning". The biggest difference between MAML and pre-training weights：Pre-training weights minimize only for original task loss. MAML can minimize all task loss with a few steps of training.

If this works for you, please give me a star, this is very important to me.😊
## Part 2. Quick  Start

1. Pull repository.

```shell
git clone https://github.com/Runist/MAML-keras.git
```

2. You need to install some dependency package.

```shell
cd MAML-keras
pip install -r requirements.txt
```

3. Download the *Omiglot* dataset and maml weights.

```shell
wget https://github.com/Runist/MAML-keras/releases/download/v1.0/Omniglot.tar
wget https://github.com/Runist/MAML-keras/releases/download/v1.0/maml.h5
tar -xvf Omniglot.tar
```

4. Run **train_mnist.py**, after few minutes, you'll get mnist weight.

```shell
python train_mnist.py
```

```
235/235 [==============================] - 62s 133ms/step - loss: 0.3736 - sparse_categorical_accuracy: 0.8918
Epoch 2/3
235/235 [==============================] - 2s 9ms/step - loss: 0.0385 - sparse_categorical_accuracy: 0.9886
Epoch 3/3
235/235 [==============================] - 2s 9ms/step - loss: 0.0219 - sparse_categorical_accuracy: 0.9934
313/313 [==============================] - 27s 48ms/step - loss: 0.0373 - sparse_categorical_accuracy: 0.9882
```

5. Run **evaluate.py**, you'll see the difference between MAML and MNIST initialization weights.

```shell
python evaluate.py
```

```
Model with mnist initialize weight train for 3 step, val loss: 1.8765, accuracy: 0.3400.
Model with mnist initialize weight train for 5 step, val loss: 1.5195, accuracy: 0.4600.
Model with maml weight train for 3 step, val loss: 0.8904, accuracy: 0.6700.
Model with maml weight train for 5 step, val loss: 0.5034, accuracy: 0.7800.
```

## Part 3. Train your own dataset
1. You should set same parameters in **config.py**. More detail you can get in my [blog](https://blog.csdn.net/weixin_42392454/article/details/109891791?spm=1001.2014.3001.5501).

```python
parser.add_argument('--n_way', type=int, default=10,
                    help='The number of class of every task.')
parser.add_argument('--k_shot', type=int, default=1,
                    help='The number of support set image for every task.')
parser.add_argument('--q_query', type=int, default=1,
                    help='The number of query set image for every task.')
parser.add_argument('--input_shape', type=tuple, default=(28, 28, 1),
                    help='The image shape of model input.')
```

2. Start training.

```shell
python train.py --n_way=5 --k_shot=1 --q_query=1
```

## Part 4. Paper and other implement

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)
- [cbfinn/*maml*](https://github.com/cbfinn/maml)
- [dragen1860/*MAML*-Pytorch](https://github.com/dragen1860/MAML-Pytorch)
