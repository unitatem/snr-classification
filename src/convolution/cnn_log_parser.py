import re


def main():
    matchers = [
        re.compile(
            "INFO:root:Building CNN layers:(.*), activation:(.*), "
            "bottleneck_layers: (.*), dropout: (.*)\n"),
        re.compile("INFO:root:Train model: {loss_fun:(.*)}\n"),
        re.compile("INFO:root:loss:(.*)\n"),
        re.compile("INFO:root:top_1_accuracy:(.*)\n"),
        re.compile("INFO:root:top_5_accuracy:(.*)\n")]

    dataset = list()
    with open('cnn.log', 'r') as log:
        record = {}
        for i, line in enumerate(log):
            line_type = i % len(matchers)

            result = matchers[line_type].findall(line)
            if len(result) != 1:
                raise RuntimeError("Bad log file")
            result = result[0]

            if line_type == 0:
                record["layers"] = int(result[0])
                record["activation"] = result[1]
                record["bottleneck_layers"] = int(result[2])
                record["dropout"] = result[3] == "True"
            elif line_type == 1:
                record["loss_fun"] = result
            elif line_type == 2:
                record["loss"] = float(result)
            elif line_type == 3:
                record["top_1_accuracy"] = float(result)
            elif line_type == 4:
                record["top_5_accuracy"] = float(result)

                dataset.append(record)
                print(record)
                record = {}

    return dataset


if __name__ == '__main__':
    main()
