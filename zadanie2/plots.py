import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import cycle
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
     corrected_values_normalised_dataframes[name] = normaliseTestDataOutput(df.copy(deep=True), output_scalers)
     MSE_test[name] = mean_squared_error(corrected_values_normalised_dataframes[name], normalised_test_data[["real_x", "real_y"]])

# 1) WYKRES 1 (training_errors and epochs) -> x - epochs; y - training_errors; + każdemu z wariantów sieci powinna odpowiadać linia o innym kolorze
palette = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(12, 6))
for name, df in MSE_dataframes.items():
    epochs = range(1, len(df["training_errors"]) + 1)
    label = name.replace("MSE_", "")
    color = next(palette)
    plt.plot(epochs, df["training_errors"], label=label, color=color)
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
palette = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(12, 6))
for name, df in MSE_dataframes.items():
    epochs = range(1, len(df["test_errors"]) + 1)
    label = name.replace("MSE_", "")
    mse_reference = MSE_test[name.replace("MSE_", "corr_values_")]
    color = next(palette)
    plt.plot(epochs, df["test_errors"], label=label, color=color)
    plt.axhline(y=mse_reference, color=color, linestyle='--')
plt.xlabel("Epoka")
plt.ylabel("Błąd MSE")
plt.title("Porównanie błędu MSE w czasie testowania dla różnych wariantów sieci")
plt.legend(title="Wariant sieci")
ax = plt.gca()  # pobierz aktualną oś
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # wymuszenie całkowitych wartości na osi X
plt.grid(True)
plt.tight_layout()
plt.show()

def calculate_cdf(errors):
    sorted_errors = np.sort(errors) # sortowanie błędów rosnąco (jaki błąd np. 0,1; 0,5)
    cdf = np.arange(1, len(errors) + 1) / len(errors) # wartości CDF (jaki procent próbek ma taki błąd np. 0,25 -> 25 %)
    return sorted_errors, cdf

cdf_data = {}

for name, df in corrected_values_dataframes.items():
     real = test_data[["real_x", "real_y"]]
     errors = np.linalg.norm(df.values - real.values, axis=1) # oblicza odległość pomiędzy przewidywaną, a rzeczywistą pozycją dla każdego punktu (norma Euklidesowa)
     sorted_errors, cdf = calculate_cdf(errors)
     cdf_data[name] = (sorted_errors, cdf)

# dystrybuanta błędów odpowiadająca wszystkim wynikom pomiarów dynamicznych (dane z plików)
measured = test_data[["measured_x", "measured_y"]].values
real = test_data[["real_x", "real_y"]].values
baseline_errors = np.linalg.norm(measured - real, axis=1)  # oblicza odległość pomiędzy przewidywaną, a rzeczywistą pozycją dla każdego punktu (norma Euklidesowa)
baseline_sorted, baseline_cdf = calculate_cdf(baseline_errors)

# 3) WYKRES 3 - dystrybuanty błędów wyznaczone dla skorygowanych wartości wszystkich wyników pomiarów dynamicznych
palette = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(12, 6))
for name in sorted(cdf_data):
    color = next(palette)
    label = name.replace("corr_values_", "")
    sorted_errors, cdf = cdf_data[name]
    plt.plot(sorted_errors, cdf, label=label, color=color)

plt.plot(baseline_sorted, baseline_cdf, linestyle='--', color='black', label='Pomiary dynamiczne')

plt.xlabel("Błąd (mm)")
plt.ylabel("Prawdopodobieństwo skumulowane (CDF)")
plt.title("Dystrybuanta błędów predykcji dla różnych wariantów sieci")
plt.legend(title="Wariant sieci")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) WYKRES 4 - skorygowane wartości wszystkich wyników pomiarów dynamicznych uzyskane przez ten spośród wybranych wariantów sieci, który wykazał się największą skutecznością korygowania błędów

