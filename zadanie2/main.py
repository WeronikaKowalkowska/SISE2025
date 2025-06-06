import sys
import glob
import json

import torch
import torch.nn as nn
from torch.optim import Adam as TorchAdam

from func import *


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py config_*.json")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = json.load(f)

    hidden_channel_neuron_count = int(config["hidden_channel_neuron_count"])
    activation_function_letter = config["activation_function"]
    learning_rate = float(config["learning_rate"])
    stop_criterion = int(config["stop_criterion"])

    training_data = pd.DataFrame()
    test_data = pd.DataFrame()

    training_data_f8_files = glob.glob(
        "./dane/f8/stat/f8_stat_*.csv")

    training_data_f8 = pd.concat(
        [pd.read_csv(f, header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y']) for f in
         training_data_f8_files])

    training_data_f10_files = glob.glob("./dane/f10/stat/f10_stat_*.csv")
    training_data_f10 = pd.concat(
        [pd.read_csv(f, header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y']) for f in
         training_data_f10_files])

    test_data_f8_files = glob.glob("./dane/f8/dyn/f8_dyn_*.csv")
    test_data_f8 = pd.concat(
        [pd.read_csv(f, header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y']) for f in
         test_data_f8_files])

    test_data_f10_files = glob.glob("./dane/f10/dyn/f10_dyn_*.csv")
    test_data_f10 = pd.concat(
        [pd.read_csv(f, header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y']) for f in
         test_data_f10_files])

    training_data = pd.concat([training_data_f8, training_data_f10])
    test_data = pd.concat([test_data_f8, test_data_f10])

    normalised_training_data = training_data.copy(deep=True)
    normalised_test_data = test_data.copy(deep=True)

    normalised_training_data, input_scalers, output_scalers = normaliseTrainingData(normalised_training_data)
    normalised_test_data = normaliseTestData(normalised_test_data, input_scalers, output_scalers)

    activation = []
    in_channels_count = 2  # const
    out_channels_count = 2  # const
    hidden_channels_list = [hidden_channel_neuron_count]

    if (activation_function_letter == "ReLu"):
        activation = nn.ReLU
    elif (activation_function_letter == "Tanh"):
        activation = nn.Tanh
    elif (activation_function_letter == "Sigmoid"):
        activation = nn.Sigmoid

    layers = []
    input_size = in_channels_count

    for hidden_size in hidden_channels_list:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation())
        input_size = hidden_size

    layers.append(nn.Linear(input_size, out_channels_count))

    multilayer_perceptron = nn.Sequential(*layers)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(multilayer_perceptron.parameters(), lr=learning_rate)

    epochs_loss_training = []
    epochs_loss_test = []

    for epoch in range(stop_criterion):
        multilayer_perceptron.train()

        input_tensor = torch.tensor(
            normalised_training_data[["measured_x", "measured_y"]].values, dtype=torch.float32
        )
        target_tensor = torch.tensor(
            normalised_training_data[["real_x", "real_y"]].values, dtype=torch.float32
        )

        output = multilayer_perceptron(input_tensor)
        loss = loss_fn(output, target_tensor)

        epochs_loss_training.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        multilayer_perceptron.eval()

        with torch.no_grad():
            input_test = torch.tensor(
                normalised_test_data[["measured_x", "measured_y"]].values, dtype=torch.float32
            )
            target_test = torch.tensor(
                normalised_test_data[["real_x", "real_y"]].values, dtype=torch.float32
            )
            output_test = multilayer_perceptron(input_test)
            loss_test = loss_fn(output_test, target_test)
            epochs_loss_test.append(loss_test.item())

    multilayer_perceptron.eval()

    with torch.no_grad():
        input_test = torch.tensor(
            normalised_test_data[["measured_x", "measured_y"]].values, dtype=torch.float32
        )
        output_test = multilayer_perceptron(input_test)
        output_np = output_test.numpy()

    learning_result = deNormaliseTestData(output_np, output_scalers)
    learning_result_x = learning_result["real_x"]
    learning_result_y = learning_result["real_y"]

    filename = ('MSE_' + config_file + '_' + str(
        hidden_channel_neuron_count) + '_' + activation_function_letter + '_' + str(
        learning_rate) + '_' + str(stop_criterion) + '.csv').replace(".json", "").replace("config_", "")
    with open(filename, 'w') as f:
        for epoch in range(stop_criterion):
            f.writelines(str(epochs_loss_training[epoch]) + ',' + str(epochs_loss_test[epoch]) + '\n')

    filename = ('corr_values_' + config_file + '_' + str(
        hidden_channel_neuron_count) + '_' + activation_function_letter + '_' + str(
        learning_rate) + '_' + str(stop_criterion) + '.csv').replace(".json", "").replace("config_", "")
    with open(filename, 'w') as f:
        for i in range(len(learning_result_x)):
            f.writelines(str(learning_result_x[i]) + ',' + str(learning_result_y[i]) + '\n')


if __name__ == "__main__":
    main()
