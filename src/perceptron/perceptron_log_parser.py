import csv
import re


if __name__ == '__main__':
    with open('perceptron_log.csv', 'w', newline='') as parsed_log:
        writer = csv.writer(parsed_log, delimiter=';')
        writer.writerow(['clusters', 'layers', 'neurons', 'activation', 'loss', 'top_1_acc', 'top_5_acc'])

        with open('perceptron.log', 'r') as log:
            line_re = [re.compile('INFO:root:Building perceptron: '
                                  '\[(.*) clusters_cnt, '
                                  '(.*) layers, '
                                  '(.*) neurons_in_every_layer, '
                                  '(.*) activation\]'),
                       re.compile('INFO:root:loss: (.*)\n'),
                       re.compile('INFO:root:top_1_accuracy: (.*)\n'),
                       re.compile('INFO:root:top_5_accuracy: (.*)\n')]

            row = []
            for i, line in enumerate(log):
                line_type = i % len(line_re)
                result = line_re[line_type].findall(line)
                if isinstance(result[0], tuple):
                    result = list(result[0])
                row = row + result
                if line_type == len(line_re) - 1:
                    writer.writerow(row)
                    row = []
