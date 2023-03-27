// See https://aka.ms/new-console-template for more information
using NNFun;

List<int> hiddenLayers = new()
{
    20, 20
};
List<DataSet> data = MnistReader.GetData();

NeuralNetConfig config = new(
    hiddenLayers,
    data,
    new Sigmoid(),
    new SoftMax()
);

NeuralNet neuralNet = new(config);

neuralNet.RunNeuralNet();