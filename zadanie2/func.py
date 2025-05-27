from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def normaliseTrainingData(df):
    input_scalers = {
        "measured_x": MinMaxScaler(),
        "measured_y": MinMaxScaler()
    }
    output_scalers = {
        "real_x": MinMaxScaler(),
        "real_y": MinMaxScaler()
    }

    normalised_input_x_array = input_scalers["measured_x"].fit_transform(df[["measured_x"]])
    normalised_input_y_array = input_scalers["measured_y"].fit_transform(df[["measured_y"]])
    normalised_output_x_array = output_scalers["real_x"].fit_transform(df[["real_x"]])
    normalised_output_y_array = output_scalers["real_y"].fit_transform(df[["real_y"]])

    normalised_df = pd.DataFrame({
        "measured_x": normalised_input_x_array.flatten(),
        "measured_y": normalised_input_y_array.flatten(),
        "real_x": normalised_output_x_array.flatten(),
        "real_y": normalised_output_y_array.flatten()
    })
    return normalised_df, input_scalers, output_scalers


def normaliseTestData(df, input_scalers, output_scalers):
    normalised_input_x_array = input_scalers["measured_x"].transform(
        df[["measured_x"]])  # transformuje dane testowe tym samym skalarem
    normalised_input_y_array = input_scalers["measured_y"].transform(df[["measured_y"]])
    normalised_output_x_array = output_scalers["real_x"].transform(df[["real_x"]])
    normalised_output_y_array = output_scalers["real_y"].transform(df[["real_y"]])

    normalised_df = pd.DataFrame({
        "measured_x": normalised_input_x_array.flatten(),
        "measured_y": normalised_input_y_array.flatten(),
        "real_x": normalised_output_x_array.flatten(),
        "real_y": normalised_output_y_array.flatten()
    })
    return normalised_df


def deNormaliseTestData(normalised_array, output_scalers):
    denormalised_array_x = output_scalers["real_x"].inverse_transform(normalised_array[:, 0].reshape(-1, 1))
    denormalised_array_y = output_scalers["real_y"].inverse_transform(normalised_array[:, 1].reshape(-1, 1))
    denormalised_df = pd.DataFrame({
        "real_x": denormalised_array_x.flatten(),
        "real_y": denormalised_array_y.flatten()
    })
    return denormalised_df
