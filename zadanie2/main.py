from torchvision.ops import MLP

def normaliseTrainingData(what):
    xMin=min(what)
    xMax=max(what)
    for i in range(len(what)):
        what[i]=(what[i]-xMin)/(xMax-xMin)
    return what, xMin, xMax

def normaliseTestData(what, xMin, xMax):
    for i in range(len(what)):
        what[i]=(what[i]-xMin)/(xMax-xMin)
    return what

training_data = []
test_data = []

in_channels_count = 2 #const
hidden_channels_count = 4
# the last value in the hidden_channels list is treated as the output size
out_channels_count = 2 #const
hidden_channels_list = [hidden_channels_count, out_channels_count]

# funkcji logistycznej, tangensu hiperbolicznego, jednostronnie obciętej funkcji liniowej (ReLU)
activation_function = []

# parametrów procesu uczenia (takich jak wartości odpowiednich współczynników czy kryterium stopu)

#https://docs.pytorch.org/vision/master/generated/torchvision.ops.MLP.html
multilayer_perceptron = MLP(in_channels=in_channels_count, hidden_channels=hidden_channels_list)