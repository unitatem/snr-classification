import config
import matplotlib.pyplot as plt
import pandas as pd

from __future__ import unicode_literals


def load_dataset(df, **kwargs):
    possible_variables = {'activation', 'layers', 'bottleneck_layers', 'loss_fun'}
    variable = list(possible_variables - set(list(kwargs.keys())))

    # casting to proper data types
    df['top_1_accuracy'] = df['top_1_accuracy'].astype('float64')
    df['top_5_accuracy'] = df['top_5_accuracy'].astype('float64')
    df['loss'] = df['loss'].astype('float64')
    df['layers'] = df['layers'].astype('int32')
    df['bottleneck_layers'] = df['bottleneck_layers'].astype('int32')

    fun_names ={variable[0]: set(df[variable[0]])}  # possible variable parameter values
    db = (df.loc[(df[list(kwargs.keys())[0]] == list(kwargs.values())[0]) & (df[list(kwargs.keys())[1]] == list(kwargs.values())[1]) & (df[list(kwargs.keys())[2]] == list(kwargs.values())[2])])

    return db, fun_names


def plot_dataset(data, keys, top_n_accuracy, variable_key):
    for activation, db in data.items():
        plot_function(db, activation, keys, top_n_accuracy, variable_key)


def plot_function(data, activation, keys, top_n_accuracy, variable_key):
    # plt.plot(data['neurons'], data[key], 'x--', label=activation)
    plt.plot(list(keys[activation][variable_key]), data[top_n_accuracy], 'x--', label=activation)


def decorate_plot(ax, ylabel, ylim, variable):
    ax.set_xlabel(variable)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend(loc=2)
    ax.grid()


def plot_save_and_close(fig, name):
    plt.subplots_adjust(wspace=0.35)

    plt.savefig(name)
    plt.show()
    plt.close(fig)

def aggregate_into_plot(df, **kwargs):
    distinct_activation_funcs = set(df['activation'])
    distinct_layer_nr = set(df['layers'])
    distinct_bottleneck_layers = set(df['bottleneck_layers'])
    distinct_loss_funcs = set(df['loss_fun'])
    dbs = {}
    func_names = {}
    for i in distinct_activation_funcs:
        # print(i)
        kwargs.update({'activation': i})
        # kwargs = {'activation': i, 'bottleneck_layers': 128, 'loss_fun': 'mean_squared_error' }
        db, fun_names = load_dataset(df, **kwargs)
        dbs[i] = db
        func_names[i] = fun_names

    return dbs, func_names



def main(neurons, loss, variable):
    filename = "cnn_results.csv"
    keys = ['dropout', 'activation', 'top_1_accuracy', 'layers', 'loss', 'top_5_accuracy', 'bottleneck_layers', 'loss_fun']

    df = pd.read_csv(filename, delimiter=",", index_col=False)
    df = df.drop(['dropout'], axis=1)

    kwargs = {'bottleneck_layers': neurons, 'loss_fun': loss}
    dbs, funcs_names = aggregate_into_plot(df, **kwargs)

    fig = plt.figure()
    plt.suptitle("Bottleneck neurons: {}, loss_func: {}".format(neurons, loss))
    top1_subplot = fig.add_subplot(1, 2, 1)
    plot_dataset(dbs, funcs_names, 'top_1_accuracy', variable)
    decorate_plot(top1_subplot, 'top_1_acc', [0.0, 0.3], variable)

    top5_subplot = fig.add_subplot(1, 2, 2)
    plot_dataset(dbs, funcs_names, 'top_5_accuracy', variable)
    decorate_plot(top5_subplot, 'top_5_acc', [0.0, 1.0], variable)

    plot_save_and_close(fig, "activation_bottleneck_" + str(neurons) + "_" + loss + ".jpg")

# 3 różne rozmiary deskryptora
# - liczba warstw splotowych
# - różne rodzaje funkcji kosztów
# - różne funkcje aktywacji


if __name__ == "__main__":

    for neurons in config.bottleneck_layer_sizes:
        for loss in config.loss_functions:
            main(neurons, loss, 'layers')
