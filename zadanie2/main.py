import sys
import glob
import json
from fileinput import filename

import pandas as pd
import numpy as np
from collections import OrderedDict

from pandas.core.interchange.dataframe_protocol import DataFrame
from scipy.io.wavfile import write
from torchvision.ops import MLP
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

    # TRENING - STAT
    # TEST - DYN
    training_data_f8_files = glob.glob(
        "./dane/f8/stat/f8_stat_*.csv")  # zwraca listę plików o nazwie pasującej do wzoru

    training_data_f8 = pd.concat(
        [pd.read_csv(f, header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y']) for f in
         training_data_f8_files])  # łączy dane z wielu plików

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

    # test_data.to_csv("test_data.csv", index=False, header=False)

    normalised_training_data = training_data.copy(deep=True)
    normalised_test_data = test_data.copy(deep=True)

    normalised_training_data, input_scalers, output_scalers = normaliseTrainingData(normalised_training_data)
    normalised_test_data = normaliseTestData(normalised_test_data, input_scalers, output_scalers)

    # normalised_test_data.to_csv("normalised_test_data.csv", index=False, header=False)

    activation = []
    in_channels_count = 2  # const
    out_channels_count = 2  # const
    hidden_channels_list = [hidden_channel_neuron_count]

    # funkcji logistycznej, tangensu hiperbolicznego, jednostronnie obciętej funkcji liniowej (ReLU)
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
        layers.append(activation())  # dodajemy aktywację po każdej warstwie
        input_size = hidden_size

    # warstwa wyjściowa (bez aktywacji)
    layers.append(nn.Linear(input_size, out_channels_count))

    # https://docs.pytorch.org/vision/master/generated/torchvision.ops.MLP.html
    multilayer_perceptron = nn.Sequential(*layers)

    # metoda optymalizacji - Adam
    # https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(multilayer_perceptron.parameters(), lr=learning_rate)

    epochs_loss_training = []
    epochs_loss_test = []

    # trenowanie
    for epoch in range(stop_criterion):
        multilayer_perceptron.train()

        input_tensor = torch.tensor(  # tworzy macierz o wymiarach [measured_x, measured_y]
            normalised_training_data[["measured_x", "measured_y"]].values, dtype=torch.float32
            # .values konwertuje pandas.DataFrame do numpy.ndarray

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

        # testowanie (bez gradientów)
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

    # denormalizowanie danych
    learning_result = deNormaliseTestData(output_np, output_scalers)
    learning_result_x = learning_result["real_x"]
    learning_result_y = learning_result["real_y"]

    # zapisywanie do plików csv
    filename = ('MSE_' + config_file + '_' + str(
        hidden_channel_neuron_count) + '_' + activation_function_letter + '_' + str(
        learning_rate) + '_' + str(stop_criterion) + '.csv').replace(".json", "").replace("config_", "")
    with open(filename, 'w') as f:
        for epoch in range(stop_criterion):
            f.writelines(str(epochs_loss_training[epoch]) + ',' + str(epochs_loss_test[epoch]) + '\n')

    filename = ('corr_values_' + config_file + '_' + str(
        hidden_channel_neuron_count) + '_' + activation_function_letter + '_' + str(
        learning_rate) + '_' + str(stop_criterion) + '_best' + '.csv').replace(".json", "").replace("config_", "")
    with open(filename, 'w') as f:
        for i in range(len(learning_result_x)):
            f.writelines(str(learning_result_x[i]) + ',' + str(learning_result_y[i]) + '\n')

    # zapisywanie perceptronu, do łatwego odtworzenia poprzez: torch.load("filename")
    torch.save(multilayer_perceptron, ("multilayer_perceptron_" + config_file + '_' + str(
        hidden_channel_neuron_count) + '_' + activation_function_letter + '_' + str(learning_rate) + '_' + str(
        stop_criterion)).replace(".json", "").replace("config_", ""))


if __name__ == "__main__":
    main()
