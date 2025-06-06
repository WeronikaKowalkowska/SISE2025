import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import cycle
from sklearn.metrics import mean_squared_error
import glob

test_data = pd.read_csv("test_data.csv", header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y'])
normalised_test_data = pd.read_csv("normalised_test_data.csv", header=None, names=['measured_x', 'measured_y', 'real_x', 'real_y'])

MSE_files = glob.glob("./MSE_*_best.csv")
MSE_dataframes = {}

for file_path in MSE_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path, header=None, names=["training_errors", "test_errors"])
    MSE_dataframes[base_name] = df

corrected_values_files = glob.glob("./corr_values_*_best.csv")
corrected_values_dataframes = {}

for file_path in corrected_values_files:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path, header=None, names=["real_x", "real_y"])
    corrected_values_dataframes[base_name] = df

# 1) WYKRES 1 (training_errors and epochs) -> x - epochs; y - training_errors; + każdemu z wariantów sieci powinna odpowiadać linia o innym kolorze
palette = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(16, 6))
for name, df in MSE_dataframes.items():
    epochs = range(1, len(df["training_errors"]) + 1)
    label = name.replace("MSE_", "").replace("_best", "")
    label = label.split('_', 1)[1]
    label = f"Wariant sieci: {label}"
    color = next(palette)
    plt.plot(epochs, df["training_errors"], label=label, color=color)
plt.xlabel("Epoka", fontsize=14)
plt.ylabel("Błąd MSE", fontsize=14)
plt.title("Porównanie błędu MSE w czasie uczenia dla różnych wariantów sieci", fontsize=16)
plt.legend()
ax = plt.gca()  # pobierz aktualną oś
ax.set_yscale("log")
ax.set_xlim(left=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # wymuszenie całkowitych wartości na osi X
plt.grid(True)
plt.tight_layout()
#plt.savefig("plot_1.png", dpi=300)

# all_last_errors = [df["training_errors"][-10:].max() for df in MSE_dataframes.values()]
# max_visible_error = max(all_last_errors) * 1.1  # trochę marginesu
# ax.set_ylim(top=max_visible_error)

plt.show()

MSE_test = mean_squared_error(normalised_test_data[["measured_x", "measured_y"]], normalised_test_data[["real_x", "real_y"]])

# 2) WYKRES 2 (test_errors and epochs) -> x - epochs; y - test_errors; + pozioma linia ciągnąca się przez całą szerokość wykresu na wysokości odpowiadającej wartości błędu średniokwadratowego wyznaczonego dla zmierzonych wartości występujących w zbiorze testowym (PRZESKALOWAĆ)
palette = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(16, 6))
for name, df in MSE_dataframes.items():
    epochs = range(1, len(df["test_errors"]) + 1)
    label = name.replace("MSE_", "").replace("_best", "")
    label = label.split('_', 1)[1]
    label = f"Wariant sieci: {label}"
    color = next(palette)
    plt.plot(epochs, df["test_errors"], label=label, color=color)
plt.axhline(y=MSE_test, color='hotpink', linestyle='--',  label='Pomiary dynamiczne')
plt.xlabel("Epoka", fontsize=14)
plt.ylabel("Błąd MSE",fontsize=14)
plt.title("Porównanie błędu MSE w czasie testowania dla różnych wariantów sieci", fontsize=16)
plt.legend()
ax = plt.gca()  # pobierz aktualną oś
ax.set_yscale("log")
ax.set_xlim(left=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # wymuszenie całkowitych wartości na osi X
plt.grid(True)
plt.tight_layout()
#plt.savefig("plot_2.png", dpi=300)

# all_last_test_errors = [df["test_errors"][-10:].max() for df in MSE_dataframes.values()]
# max_visible_test_error = max(all_last_test_errors) * 1.1
# ax.set_ylim(top=max_visible_test_error)

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
plt.figure(figsize=(16, 6))
for name in sorted(cdf_data):
    color = next(palette)
    label = name.replace("corr_values_", " ").replace("_best", "")
    label = label.split('_', 1)[1]
    label = f"Wariant sieci: {label}"
    sorted_errors, cdf = cdf_data[name]
    plt.plot(sorted_errors, cdf, label=label, color=color)

plt.plot(baseline_sorted, baseline_cdf, linestyle='--', color='hotpink', label='Pomiary dynamiczne')

plt.xlabel("Błąd (mm)", fontsize=14)
plt.ylabel("Prawdopodobieństwo skumulowane (CDF)", fontsize=14)
plt.title("Dystrybuanta błędów predykcji dla różnych wariantów sieci", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("plot_3.png", dpi=300)
plt.show()
# 4) WYKRES 4 - skorygowane wartości wszystkich wyników pomiarów dynamicznych uzyskane przez ten spośród wybranych wariantów sieci, który wykazał się największą skutecznością korygowania błędów
measured_corr = {}

palette = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(16, 6))

plt.scatter(test_data["measured_x"],test_data["measured_y"], color='pink', label="Wartości zmierzone")

for name, df in corrected_values_dataframes.items():
    measured_corr[name] = df[["real_x", "real_y"]]
    label = name.replace("corr_values_", "").replace("_best", "")
    label = label.split('_', 1)[1]
    label = f"Wariant sieci: {label}"
    color = next(palette)
    plt.scatter(df["real_x"], df["real_y"], color=color, label=label)

plt.scatter(test_data["real_x"],test_data["real_y"], color='blue', label="Wartości rzeczywiste")

plt.xlabel("Wartość x", fontsize=14)
plt.ylabel("Wartość y", fontsize=14)
plt.title("Porównanie wyników pomiarów dynamicznych dla najlepszych wariantów sieci", fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("plot_4.png", dpi=300)
plt.show()
