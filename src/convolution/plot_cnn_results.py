from __future__ import unicode_literals

import config
import matplotlib.pyplot as plt
import pandas as pd

FILENAME_RESULTS = "cnn_results.csv"


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

    sings = ['x--', 'x--', 'x--', 'x--']
    i = 0
    for activation, db in data.items():
        plot_function(db, activation, keys, top_n_accuracy, variable_key, sings[i])
        i += 1


def plot_function(data, activation, keys, top_n_accuracy, variable_key, sign):
    plt.plot(list(keys[activation][variable_key]), data[top_n_accuracy], sign, label=activation)


def decorate_plot(ax, ylabel, ylim, variable):

    if variable == "bottleneck_layers":
        variable = "bottleneck_neurons"
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
    dbs = {}
    func_names = {}
    for i in distinct_activation_funcs:
        kwargs.update({'activation': i})
        db, fun_names = load_dataset(df, **kwargs)
        dbs[i] = db
        func_names[i] = fun_names

    return dbs, func_names


def main_variable_layers(neurons, loss, variable):

    df = pd.read_csv(FILENAME_RESULTS, delimiter=",", index_col=False)
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


def main_variable_neurons(layers, loss, variable):

    df = pd.read_csv(FILENAME_RESULTS, delimiter=",", index_col=False)
    df = df.drop(['dropout'], axis=1)

    kwargs = {'layers': layers, 'loss_fun': loss}
    dbs, funcs_names = aggregate_into_plot(df, **kwargs)

    fig = plt.figure()
    plt.suptitle("Layers: {}, loss_func: {}".format(layers, loss))
    top1_subplot = fig.add_subplot(1, 2, 1)
    plot_dataset(dbs, funcs_names, 'top_1_accuracy', variable)
    decorate_plot(top1_subplot, 'top_1_acc', [0.0, 0.4], variable)

    top5_subplot = fig.add_subplot(1, 2, 2)
    plot_dataset(dbs, funcs_names, 'top_5_accuracy', variable)
    decorate_plot(top5_subplot, 'top_5_acc', [0.0, 1.0], variable)

    plot_save_and_close(fig, "activation_layers_" + str(layers) + "_" + loss + ".jpg")


def calculate_stats():
    df = pd.read_csv(FILENAME_RESULTS, delimiter=",", index_col=False)
    df = df.drop(['dropout'], axis=1)  # redundant column
    new_df = df[df['layers'] <= 3]  # reduced only to 3 cnn layers

    stats_dict_4_layers = {'avg': df.mean(0), 'max': df.max(0), 'std': df.std(0), 'min': df.min(0)}
    stats_dict_3_layers = {'avg': new_df.mean(0), 'max': new_df.max(0), 'std': new_df.std(0), 'min': new_df.min(0)}

    print("--------------------- 4 CNN - STATS ------------------------------")
    print(stats_dict_4_layers)
    print("--------------------- 3 CNN - STATS ------------------------------")
    print(stats_dict_3_layers)
    print("--------------------- MAX ------------------------------")
    print(new_df.loc[new_df['top_1_accuracy'] == new_df['top_1_accuracy'].max()])
    print("--------------------- MIN ------------------------------")
    print(new_df.loc[new_df['top_1_accuracy'] == new_df['top_1_accuracy'].min()])


if __name__ == "__main__":
    show_stats = True
    show_bottleneck = False
    show_cnn_layers = False

    if show_stats:
        calculate_stats()

    if show_bottleneck:
        for neurons in config.bottleneck_layer_sizes:
            for loss in config.loss_functions:
                main_variable_layers(neurons, loss, 'layers')

    if show_cnn_layers:
        for layer_cnt in range(config.layer_cnt_start, config.layer_cnt_stop + 1, config.layer_cnt_step):
            for loss in config.loss_functions:
                main_variable_neurons(layer_cnt, loss, 'bottleneck_layers')
