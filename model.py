from torch import nn


def build_classifier(input_size, output_size, hidden_layers, drop_p=0.2):

    """ Builds a feed forward network with arbitraty hidden layers:
    Args:
    num_in_features: integer, size of the input layer
    num_out_features: integer, size of the output layer
    hidden_layers: list of integers, the size of the hidden layers

    Example build_classifier(784, 2, [512, 256, 128, 64, 32, 16, 8])
    """

    classifier = nn.Sequential()
    if len(hidden_layers) == 0 or hidden_layers == None:
        classifier.add_module("fc0", nn.Linear(input_size, 102))
        classifier.add_module("output", nn.LogSoftmax(dim=1))
    else:
        classifier.add_module("fc0", nn.Linear(input_size, hidden_layers[0]))
        classifier.add_module("relu0", nn.ReLU())
        classifier.add_module("drop0", nn.Dropout(p=drop_p))

        layer_size = zip(hidden_layers[:-1], hidden_layers[1:])
        for index, (in_, out) in enumerate(layer_size):
            classifier.add_module(f"fc{index+1}", nn.Linear(in_, out))
            classifier.add_module(f"relu{index+1}", nn.ReLU())
            classifier.add_module(f"drop{index+1}", nn.Dropout(p=drop_p))
        else:
            classifier.add_module(
                f"fc{len(hidden_layers) + 1}",
                nn.Linear(hidden_layers[-1], output_size),
            )
            classifier.add_module("output", nn.LogSoftmax(dim=1))
    return classifier
