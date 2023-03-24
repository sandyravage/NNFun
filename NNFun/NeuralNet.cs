using System.Data;

namespace NNFun;

internal class NeuralNet
{
    private readonly List<int> _hiddenLayers;
    private readonly List<List<Neuron>> _layers = new();
    private readonly double _alpha;
    private readonly double _p;
    private readonly IHiddenFunction _hiddenFunction;
    private readonly IOutcomeFunction _outcomeFunction;

    private List<DataSet> _trainingData = new();
    private List<DataSet> _testingData = new();
    private DataSet _currentDataSet;
    private static readonly Random _rnd = new();

    internal NeuralNet(NeuralNetConfig config)
    {
        _hiddenLayers = config.HiddenLayers;
        RandomizeData(config.Data);
        _hiddenFunction = config.HiddenFunction;
        _outcomeFunction = config.OutcomeFunction;
        _alpha = config.DeltaCoefficient;
        _p = config.MomentumCoefficient;
    }

    public void RunNeuralNet()
    {
        Console.WriteLine("Beginning neural net run...\n");
        ValidateData();
        CreateNeurons();
        Train();
        Test();
    }
    
    private void Test()
    {
        Console.WriteLine("Beginning Testing...\n");
        foreach (DataSet dataSet in _testingData)
        {
            _currentDataSet = dataSet;
            ForwardProp();
            LogOutcome();
        }
    }

    private void Train()
    {
        Console.WriteLine("Beginning Training...\n");
        foreach (DataSet dataSet in _trainingData)
        {
            _currentDataSet = dataSet;
            ForwardProp();
            BackwardProp();
        }
        Console.WriteLine("Training Complete\n");
    }

    private void LogOutcome()
    {
        List<double> outcomes = _layers.Last().Select(x => x.Output).ToList();
        Console.WriteLine($"Network outcome: {outcomes.IndexOf(outcomes.Max())}");
        double errorRate = 0.0;
        for (int i = 0; i < outcomes.Count; i++)
        {
            errorRate += Math.Abs(_currentDataSet.Targets[i] - outcomes[i]);
        }
        Console.WriteLine($"Error rate: {errorRate}\n");
    }

    private void ValidateData()
    {
        if (!_trainingData.Any())
        {
            throw new DataInitializationException("No training data");
        }
        if (_trainingData.Select(x => x.Data.Count).Distinct().Count() > 1)
        {
            throw new DataInitializationException("Training data sizes do not match");
        }
        if (_trainingData.Select(x => x.Targets.Count).Distinct().Count() > 1)
        {
            throw new DataInitializationException("Output layer sizes do not match");
        }
    }

    private void CreateNeurons()
    {
        for (int i = 0; i < _hiddenLayers.Count; i++)
        {
            int previousInputSize = i == 0 ? _trainingData.First().Data.Count : _layers.Last().Count;
            CreateLayer(_hiddenLayers[i], previousInputSize);
        }
        CreateLayer(_trainingData.First().Targets.Count, _layers.Last().Count);  
    }

    private void CreateLayer(int layerSize, int previousInputSize)
    {
        List<Neuron> layer = new();
        for (int i = 0; i < layerSize; i++)
        {
            Neuron neuron = new(GetRandomInRange(3.0, -3.0), RandomizeWeights(previousInputSize));
            layer.Add(neuron);
        }
        _layers.Add(layer);
    }

    private void ForwardProp()
    {
        for (int i = 0; i < _layers.Count; i++)
        {
            if(i == 0)
            {
                _layers[i].ForEach(x => x.Output = _hiddenFunction.Activation(x.ComputeOutput(_currentDataSet.Data)));
            }
            else
            {
                _layers[i].ForEach(x => x.Output = _hiddenFunction.Activation(x.ComputeOutput(_layers[i - 1].Select(y => y.Output).ToList())));
            }
        }
        List<double> finalOutput = _layers[^2].Select(y => y.Output).ToList();
        _layers[^1].ForEach(x => x.Output = _outcomeFunction.Activation(finalOutput, x.ComputeOutput(finalOutput)));
    }

    private void BackwardProp()
    {
        for (int i = _layers.Count - 1; i > -1; i--)
        {
            List<double> previousOutputs = i == 0 ? _currentDataSet.Data :_layers[i - 1].Select(x => x.Output).ToList();

            if (i == _layers.Count - 1)
            {
                for (int j = 0; j < _layers[i].Count; j++)
                {
                    Neuron neuron = _layers[i][j];
                    neuron.Gradient = CalculateOutcomeGradient(neuron.Output, _currentDataSet.Targets[j]);
                    for (int k = 0; k < neuron.Weights.Count; k++)
                    {
                        double weightDelta = CalculateDelta(previousOutputs[k], neuron.Gradient);
                        neuron.Weights[k] = weightDelta + CalculateMomentum(neuron.PreviousWeightDeltas[k]);
                        neuron.PreviousWeightDeltas[k] = weightDelta;
                    }
                    double biasDelta = CalculateDelta(1, neuron.Gradient);
                    neuron.Bias = biasDelta + CalculateMomentum(neuron.PreviousBiasDelta);
                    neuron.PreviousBiasDelta = biasDelta;
                }
            }
            else
            {
                for (int j = 0; j < _layers[i].Count; j++)
                {
                    List<double> previousWeights = _layers[i + 1].Select(x => x.Weights[j]).ToList();
                    List<double> previousGradients = _layers[i + 1].Select(x => x.Gradient).ToList();
                    Neuron neuron = _layers[i][j];
                    neuron.Gradient = CalculateHiddenGradient(neuron.Output, previousGradients, previousWeights);
                    for (int k = 0; k < neuron.Weights.Count; k++)
                    {
                        double weightDelta = CalculateDelta(previousOutputs[k], neuron.Gradient);
                        neuron.Weights[k] = weightDelta + CalculateMomentum(neuron.PreviousWeightDeltas[k]);
                        neuron.PreviousWeightDeltas[k] = weightDelta;
                    }
                    double biasDelta = CalculateDelta(1, neuron.Gradient);
                    neuron.Bias = biasDelta + CalculateMomentum(neuron.PreviousBiasDelta);
                    neuron.PreviousBiasDelta = biasDelta;
                }
            }
        }
    }

    private double CalculateDelta(double previousOutput, double gradient)
    {
        return _alpha * previousOutput * gradient;
    }

    private double CalculateMomentum(double previousDelta)
    {
        return _p * previousDelta;
    }

    private double CalculateOutcomeGradient(double outcome, double desired)
    {
        return _outcomeFunction.Derivative(outcome) * (desired - outcome);
    }

    private double CalculateHiddenGradient(double outcome, List<double> previousGradients, List<double> previousWeights)
    {
        double sum = 0;
        for (int i = 0; i < previousGradients.Count; i++)
        {
            sum += previousGradients[i] * previousWeights[i];
        }
        return _hiddenFunction.Derivative(outcome) * sum;
    }

    private static List<double> RandomizeWeights(int layerSize)
    {
        
        List<double> weights = new();
        for (int i = 0; i < layerSize; i++)
        {
            weights.Add(GetRandomInRange(0.01, -0.01));
        }
        return weights;
    }

    private static double GetRandomInRange(double upper, double lower)
    {
        if (upper < lower)
        {
            throw new RangeBoundException($"Upper range param: {upper} was not greater than lower range param: {lower}");
        }
        return (_rnd.NextDouble() * (upper - lower)) + lower;
    }

    private void RandomizeData(List<DataSet> data)
    {
        int quarterSetIndex = data.Count / 4;
        _testingData = data.Take(quarterSetIndex).OrderBy(x => _rnd.Next()).ToList();
        _trainingData = data.Skip(quarterSetIndex).OrderBy(x => _rnd.Next()).ToList();
    }
}

internal class Neuron
{
    internal double Bias;
    internal List<double> Weights;
    internal double Output;
    internal double Gradient;
    internal List<double> PreviousWeightDeltas;
    internal double PreviousBiasDelta;
    internal Neuron(double bias, List<double> weights)
    {
        Bias = bias;
        Weights = weights;
        PreviousWeightDeltas = Enumerable.Repeat(0.0, weights.Count).ToList();
        PreviousBiasDelta = 0.0;
    }

    internal double ComputeOutput(List<double> inputs)
    {
        if(Weights.Count != inputs.Count)
        {
            throw new WeightsValuesMismatchException($"Weights count: {Weights.Count} does not match inputs count: {inputs.Count}");
        }
        double product = 0;
        for (int i = 0; i < inputs.Count; i++)
        {
            product += Weights[i] * inputs[i];
        }
        return product + Bias;
    }
}

public class WeightsValuesMismatchException : Exception
{
    internal WeightsValuesMismatchException(string message) : base(message)
    {

    }
}

public class RangeBoundException : Exception
{
    internal RangeBoundException(string message) : base(message)
    {

    }
}

public class DataInitializationException : Exception
{
    internal DataInitializationException(string message) : base(message)
    {

    }
}