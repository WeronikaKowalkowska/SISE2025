from sklearn.preprocessing import MinMaxScaler
import pandas as pd

input_columns = ["measured_x", "measured_y"]
output_columns = ["real_x", "real_y"]


def normaliseTrainingData(df):
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()
    normalised_input_array = input_scaler.fit_transform(df[input_columns])
    normalised_output_array = output_scaler.fit_transform(df[output_columns])
    normalised_input_df = pd.DataFrame(normalised_input_array, columns=input_columns)
    normalised_output_df = pd.DataFrame(normalised_output_array, columns=output_columns)
    normalised_df = pd.concat([normalised_input_df, normalised_output_df], axis=1)
    return normalised_df, input_scaler, output_scaler


def normaliseTestData(df, input_scaler, output_scaler):
    normalised_input_array = input_scaler.transform(df[input_columns])  # transformuje dane testowe tym samym skalarem
    normalised_output_array = output_scaler.transform(df[output_columns])
    normalised_input_df = pd.DataFrame(normalised_input_array, columns=input_columns)
    normalised_output_df = pd.DataFrame(normalised_output_array, columns=output_columns)
    normalised_df = pd.concat([normalised_input_df, normalised_output_df], axis=1)
    return normalised_df


def deNormaliseTestData(normalised_array, output_scaler):
    denormalised_array = output_scaler.inverse_transform(normalised_array)
    return denormalised_array
