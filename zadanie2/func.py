def normaliseTrainingData(what):
    xMin = what.min()
    xMax = what.max()
    for col in what.columns:
        what[col] = (what[col] - xMin[col]) / (xMax[col] - xMin[col])
    return what, xMin, xMax


def normaliseTestData(what, xMin, xMax):
    for col in what.columns:
        what[col] = (what[col] - xMin[col]) / (xMax[col] - xMin[col])
    return what


def deNormaliseTestData(what, xMin, xMax):
    denormalised = []
    for v in what:
        denorm_value = v * (xMax - xMin) + xMin  # + xMin, bo normalizacja to (x - xMin)/(xMax - xMin)
        denormalised.append(denorm_value)
    return denormalised
