using System;

namespace NNFun;

internal readonly record struct NeuralNetConfig(
    List<int> HiddenLayers,
    List<DataSet> Data,
    IHiddenFunction HiddenFunction,
    IOutcomeFunction OutcomeFunction,
    double DeltaCoefficient = 0.05,
    double MomentumCoefficient = 0.01
);
