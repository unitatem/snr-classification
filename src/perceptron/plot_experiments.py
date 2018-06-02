from __future__ import unicode_literals

import matplotlib.pyplot as plt
import pandas as pd


def load_dataset(df, cluster, layer):
    df['top_1_acc'] = df['top_1_acc'].astype('float64')
    df['top_5_acc'] = df['top_5_acc'].astype('float64')
    df['neurons'] = df['neurons'].astype('int32')

    relu = df.loc[df['activation'] == "relu"]
    elu = df.loc[df['activation'] == "elu"]
    sigmoid = df.loc[df['activation'] == "sigmoid"]
    tanh = df.loc[df['activation'] == "tanh"]

    elu_selected = elu.loc[(elu['clusters'] == cluster) & (elu['layers'] == layer)]
    relu_selected = relu.loc[(relu['clusters'] == cluster) & (relu['layers'] == layer)]
    tanh_selected = tanh.loc[(tanh['clusters'] == cluster) & (tanh['layers'] == layer)]
    sigmoid_selected = sigmoid.loc[(sigmoid['clusters'] == cluster) & (sigmoid['layers'] == layer)]

    db = [elu_selected, relu_selected, tanh_selected, sigmoid_selected]
    fun_names = ['elu', 'relu', 'tanh', 'sigmoid']

    return db, fun_names


def plot_dataset(dataset, key,  names):
    for data, name in zip(dataset, names):
        plot_function(data, key, name)


def plot_function(data, key, activation):
    plt.plot(data['neurons'], data[key], 'x--', label=activation)


def decorate_plot(ax, ylabel, ylim):
    ax.set_xlabel("neurons")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc=3)
    ax.grid()


def plot_save_and_close(fig, name):
    plt.subplots_adjust(wspace=0.35)

    plt.savefig(name)
    plt.show()
    plt.close(fig)


def main(cluster, layer):
    filename = "perceptron_log.csv"
    df = pd.read_csv(filename, delimiter=";", index_col=False)
    db, fun_names = load_dataset(df, cluster, layer)

    fig = plt.figure()
    plt.suptitle('{cluster} clusters /{layer} layers'.format(cluster=cluster, layer=layer), y=1.0)

    top1_subplot = fig.add_subplot(1, 2, 1)
    plot_dataset(db, 'top_1_acc', fun_names)
    decorate_plot(top1_subplot, 'top_1_acc', [0.0, 0.16])

    top5_subplot = fig.add_subplot(1, 2, 2)
    plot_dataset(db, 'top_5_acc', fun_names)
    decorate_plot(top5_subplot, 'top_5_acc', [0.15, 0.5])

    plot_save_and_close(fig, "fig_" + str(cluster) + "_" + str(layer) + ".jpg")


if __name__ == "__main__":
    for cluster in range(8, 20 + 1, 4):
        for layer in range(1, 4 + 1, 1):
            main(cluster, layer)
