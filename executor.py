import numpy as np
from mnist import load_mnist
from utils import *
from multilayer import *
from trainer import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

model = MultiLayer(input_size=784, hidden_size_list=[10, 10], output_size=10)

model_train = Trainer(network=model, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)

model_train.train()