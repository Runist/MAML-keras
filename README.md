# Keras  - MAML

## Part 1. Introduction

As we all know, deep learning need vast data. If you don't have this condition, you can use pre-training weights. Most of data can be fitted be pre-training weights,  but there all still some data that can't converge to the global lowest point. So it is exist one weights that can let all task get best result?

Yes, this is "Model-Agnostic Meta-Learning". The biggest difference between MAML and pre-training weights：Pre-training weights minimize only for original task loss. MAML can minimize all task loss with a few steps of training.

## Part 2. Quick  Start

1. Pull repository.

```shell
git clone https://github.com/Runist/MAML-keras.git
```

2. You need to install some dependency package.

```shell
cd MAML-keras
pip installl -r requirements.txt
```

3. Download the *Omiglot* dataset and maml weights.

```shell
wget https://github.com/Runist/MAML-keras/releases/download/v0.1/Omniglot.tar
wget https://github.com/Runist/MAML-keras/releases/download/v0.1/maml.h5
tar -xvf Omniglot.tar
```

4. Run **evaluate.py**, you'll see the difference between MAML and random initialization weights.

```shell
python evaluate.py
```

```
Model with random initialize weight train for 3 step, val loss: 1.8765, accuracy: 0.3400.
Model with random initialize weight train for 5 step, val loss: 1.5195, accuracy: 0.4600.
Model with random initialize weight train for 10 step, val loss: 1.5562, accuracy: 0.4800.
Model with maml weight train for 3 step, val loss: 0.8904, accuracy: 0.6700.
Model with maml weight train for 5 step, val loss: 0.5034, accuracy: 0.7800.
Model with maml weight train for 10 step, val loss: 0.2013, accuracy: 0.9500.
```

## Part 3. Train your own dataset
1. You should set same parameters in **config.py**. More detail you can get in my [blog](https://blog.csdn.net/weixin_42392454/article/details/109891791?spm=1001.2014.3001.5501).

```python
n_way = "number of classes"
k_shot = "number of support set"
q_query = "number of query set"
```

2. Start training.

```shell
python train.py
```

3. Running tensorboard to monitor the training process.

```shell
tensorboard --logdir=./summary
```

![tensorboard.png](https://i.loli.net/2021/04/30/KYx2FG3cpdrjSzu.png)

## Part 4. Paper and other implement

- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)
- [cbfinn/*maml*](https://github.com/cbfinn/maml)
- [dragen1860/*MAML*-Pytorch](https://github.com/dragen1860/MAML-Pytorch)
