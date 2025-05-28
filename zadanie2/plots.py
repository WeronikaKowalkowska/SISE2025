import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import glob
from func import *
from func import normaliseTestData

training_data_f8_files = glob.glob("./dane/f8/stat/f8_stat_*.csv")  # zwraca listę plików o nazwie pasującej do wzoru
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

normalised_training_data = training_data.copy(deep=True)
normalised_test_data = test_data.copy(deep=True)

normalised_training_data, input_scalers, output_scalers = normaliseTrainingData(normalised_training_data)
normalised_test_data = normaliseTestData(normalised_test_data, input_scalers, output_scalers)

MSE_files = glob.glob("./MSE_*.csv")
MSE_dataframes = {}

for file_path in MSE_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path, header=None, names=["training_errors", "test_errors"])
    MSE_dataframes[base_name] = df

corrected_values_files = glob.glob("./corr_values_*.csv")
corrected_values_dataframes = {}

for file_path in corrected_values_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path, header=None, names=["real_x", "real_y"])
    corrected_values_dataframes[base_name] = df

corrected_values_normalised_dataframes = {}
MSE_test = {}

for name, df in corrected_values_dataframes.items():
     corrected_values_normalised_dataframes[name] = normaliseTestDataOutput(df, output_scalers)
     MSE_test[name] = mean_squared_error(corrected_values_normalised_dataframes[name], normalised_test_data[["real_x", "real_y"]])

# 1) WYKRES 1 (training_errors and epochs) -> x - epochs; y - training_errors; + każdemu z wariantów sieci powinna odpowiadać linia o innym kolorze
plt.figure(figsize=(12, 6))
for name, df in MSE_dataframes.items():
    epochs = range(1, len(df["training_errors"]) + 1)
    label = name.replace("MSE_", "")
    plt.plot(epochs, df["training_errors"], label=label)
plt.xlabel("Epoka")
plt.ylabel("Błąd MSE")
plt.title("Porównanie błędu MSE w czasie uczenia dla różnych wariantów sieci")
plt.legend(title="Wariant sieci")
ax = plt.gca()  # pobierz aktualną oś
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # wymuszenie całkowitych wartości na osi X
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) WYKRES 2 (test_errors and epochs) -> x - epochs; y - test_errors; + pozioma linia ciągnąca się przez całą szerokość wykresu na wysokości odpowiadającej wartości błędu średniokwadratowego wyznaczonego dla zmierzonych wartości występujących w zbiorze testowym (PRZESKALOWAĆ)

plt.figure(figsize=(12, 6))
for name, df in MSE_dataframes.items():
    epochs = range(1, len(df["test_errors"]) + 1)
    label = name.replace("MSE_", "")
    mse_reference = MSE_test[name.replace("MSE_", "corr_values_")]
    plt.axhline(y=mse_reference, color='red', linestyle='--', label="Błąd odniesienia")
    plt.plot(epochs, df["test_errors"], label=label)
plt.xlabel("Epoka")
plt.ylabel("Błąd MSE")
plt.title("Porównanie błędu MSE w czasie uczenia dla różnych wariantów sieci")
plt.legend(title="Wariant sieci")
ax = plt.gca()  # pobierz aktualną oś
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # wymuszenie całkowitych wartości na osi X
plt.grid(True)
plt.tight_layout()
plt.show()

# 3) WYKRES 2

# 4) WYKRES 2
