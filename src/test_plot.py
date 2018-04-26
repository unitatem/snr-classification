#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import itertools
import matplotlib.pyplot as plt
import pandas as pd


def main():

    filename = "perceptron_log.csv"
    df = pd.read_csv(filename, delimiter=";", index_col=False)
    # print(df)
    df['top_1_acc'] = df['top_1_acc'].astype('float64')
    df['top_5_acc'] = df['top_5_acc'].astype('float64')
    df['neurons'] = df['neurons'].astype('int32')

    relu = df.loc[df['activation'] == "relu"]

    elu = df.loc[df['activation'] == "elu"]
    sigmoid = df.loc[df['activation'] == "sigmoid"]
    tanh = df.loc[df['activation'] == "tanh"]

    elu_selected = elu.loc[(elu['clusters'] == 12) & (elu['layers'] == 1)]
    relu_selected = relu.loc[(relu['clusters'] == 12) & (relu['layers'] == 1)]
    tanh_selected = tanh.loc[(tanh['clusters'] == 12) & (tanh['layers'] == 1)]
    sigmoid_selected = sigmoid.loc[(sigmoid['clusters'] == 12) & (sigmoid['layers'] == 1)]

    fig = plt.figure()

    plt.suptitle('{cluster}clusters /{layer} layers'.format(cluster=12, layer=1))
    first_subplot = fig.add_subplot(1, 2, 1)
        # first_subplot.set_title('top_1_acc/{cluster}clusters /{layer} layers'.format(cluster=cluster_layer[0], layer=cluster_layer[1]))
    first_subplot.set_xlabel("neurons")
    first_subplot.set_ylabel("top_1_acc")
    first_subplot.set_ylim(0.0, 0.15)
    plt.plot(elu_selected['neurons'],elu_selected['top_1_acc'])
    plt.plot(relu_selected['neurons'], relu_selected['top_1_acc'])
    plt.plot(tanh_selected['neurons'], tanh_selected['top_1_acc'])
    plt.plot(sigmoid_selected['neurons'], sigmoid_selected['top_1_acc'])

    second_subplot = fig.add_subplot(1, 2, 2)
        # second_subplot.set_title('top_5_acc/{cluster}clusters /{layer} layers'.format(cluster=cluster_layer[0], layer=cluster_layer[1]))
    second_subplot.set_xlabel("neurons")
    second_subplot.set_ylabel("top_5_acc")

    second_subplot.set_ylim(0.15, 0.4)
    plt.plot(elu_selected['neurons'],elu_selected['top_5_acc'])
    plt.plot(relu_selected['neurons'], relu_selected['top_5_acc'])
    plt.plot(tanh_selected['neurons'], tanh_selected['top_5_acc'])
    plt.plot(sigmoid_selected['neurons'], sigmoid_selected['top_5_acc'])

    first_subplot.legend(loc=3)
    first_subplot.grid()
    second_subplot.legend(loc=3)
    second_subplot.grid()


    plt.show()
    plt.close(fig)




if __name__ == "__main__":
    main()