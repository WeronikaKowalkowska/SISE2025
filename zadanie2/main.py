import sys
from torchvision.ops import MLP

def normaliseTrainingData(what):
    xMin = min(what)
    xMax = max(what)
    for i in range(len(what)):
        what[i] = (what[i] - xMin) / (xMax - xMin)
    return what, xMin, xMax


def normaliseTestData(what, xMin, xMax):
    for i in range(len(what)):
        what[i] = (what[i] - xMin) / (xMax - xMin)
    return what

def main():

    # arguments: hidden_channels_count, activation_function_letter, values_coefficients, stop_criterion
    if len(sys.argv) < 5:
        print("Not enough arguments.")
        sys.exit(1)

    hidden_channels_count = int(sys.argv[1])
    activation_function_letter = sys.argv[2]
    values_coefficients = sys.argv[3]
    stop_criterion = sys.argv[4]

    training_data = []
    test_data = []

    in_channels_count = 2 #const
    # the last value in the hidden_channels list is treated as the output size
    out_channels_count = 2 #const
    hidden_channels_list = [hidden_channels_count, out_channels_count]

    # funkcji logistycznej, tangensu hiperbolicznego, jednostronnie obciętej funkcji liniowej (ReLU)
    activation_function = []

    # parametrów procesu uczenia (takich jak wartości odpowiednich współczynników czy kryterium stopu)

    #https://docs.pytorch.org/vision/master/generated/torchvision.ops.MLP.html
    multilayer_perceptron = MLP(in_channels=in_channels_count, hidden_channels=hidden_channels_list)

if __name__ == "__main__":
    main()