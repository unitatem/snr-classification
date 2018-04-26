#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import itertools
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rcParams['text.latex.unicode'] = True
    filename = "perceptron_log.csv"
    # df = pd.read_csv(filename, names=['clusters'   ,'layers'  ,'neurons' ,'activation','loss','top_1_acc','top_5_acc'], delimiter=";")
    df = pd.read_csv(filename, delimiter=";")
    possible_clusters = set(df['clusters'])
    layers_number = set(df['layers'])
    clusters_layers_pairs = itertools.product(possible_clusters, layers_number)
    print(list(clusters_layers_pairs))

    activation_functions = set(df['activation'])
    result_dict = dict.fromkeys(list(clusters_layers_pairs), None)
    activation_dicts = dict.fromkeys(activation_functions, result_dict)

    for activation in activation_dicts.keys():
        subset_activation = df.loc[df['activation'] == activation]
        for cluster_layer in itertools.product(possible_clusters, layers_number):
            activation_dicts[activation][cluster_layer] = subset_activation.loc[
                (subset_activation['clusters'] == cluster_layer[0]) &
                (subset_activation['layers'] == cluster_layer[1])]
            cluster = cluster_layer[0]
            layer = cluster_layer[1]
            # print(activation_dicts[activation][cluster_layer])
    colors_dict = dict(zip(activation_dicts.keys(), ['red', 'yellow', 'green', 'blue']))
    print(colors_dict)



    for cluster_layer in itertools.product(possible_clusters, layers_number):
        fig = plt.figure()
        plt.suptitle('{cluster}clusters /{layer} layers'.format(cluster=cluster_layer[0], layer=cluster_layer[1]))

        first_subplot = fig.add_subplot(1, 2, 1)
        # first_subplot.set_title('top_1_acc/{cluster}clusters /{layer} layers'.format(cluster=cluster_layer[0], layer=cluster_layer[1]))
        first_subplot.set_xlabel("neurons")
        first_subplot.set_ylabel("top_1_acc")

        second_subplot = fig.add_subplot(1, 2, 2)
        # second_subplot.set_title('top_5_acc/{cluster}clusters /{layer} layers'.format(cluster=cluster_layer[0], layer=cluster_layer[1]))
        second_subplot.set_xlabel("neurons")
        second_subplot.set_ylabel("top_5_acc")

        second_subplot = fig.add_subplot(1, 2, 2)
        # second_subplot.set_title('top_5_acc/{cluster}clusters /{layer} layers'.format(cluster=cluster_layer[0], layer=cluster_layer[1]))

        for activation in activation_functions:
            print(activation)
            # print(cluster_layer, activation)
            # plt.subplot(1,2,1)

            # print(activation_dicts[activation][cluster_layer]['neurons'], activation_dicts[activation][cluster_layer]['top_1_acc'])
            first_subplot.plot(activation_dicts[activation][cluster_layer]['neurons'], activation_dicts[activation][cluster_layer]['top_1_acc'], '-.', color=colors_dict[activation], label=activation)
            second_subplot.plot(activation_dicts[activation][cluster_layer]['neurons'], activation_dicts[activation][cluster_layer]['top_5_acc'], '-.', color=colors_dict[activation], label=activation)

        first_subplot.legend(loc=1)
        first_subplot.grid()
        second_subplot.legend(loc=1)
        second_subplot.grid()
        plt.subplots_adjust(left=0.2, wspace=0.8, top=2.0)

        plt.show()
        # plt.savefig("clusters_" + str(cluster_layer[0]) + "_layers_" + str(cluster_layer[1]) + ".pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()