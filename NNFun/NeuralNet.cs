namespace NNFun;

internal class NeuralNet
{
    private List<DataSet> _trainingData = new();
    private List<DataSet> _testingData = new();
    private readonly List<List<Neuron>> _layers = new();
    private DataSet _currentDataSet;
    private static readonly Random _rnd = new();

    private int _iteration = 0;
    private readonly NeuralNetConfig _config;

    internal NeuralNet(NeuralNetConfig config)
    {
        _config = config;
        RandomizeData(config.Data);
    }

    public void RunNeuralNet()
    {
        Console.WriteLine("Beginning Neural Net run...\n");
        ValidateData();
        CreateNeurons();
        Train();
        Test();
        Console.WriteLine("Run Complete");
    }
    
    private void Test()
    {
        Console.WriteLine("Beginning Testing...\n");
        foreach (DataSet dataSet in _testingData)
        {
            _iteration++;
            _currentDataSet = dataSet;
            ForwardProp();
            if(_iteration % 500 == 0)
            {
                LogOutcome();
            }
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
        var indexOutput = outcomes.IndexOf(outcomes.Max());
        Console.WriteLine($"Network outcome: {outcomes.IndexOf(outcomes.Max())}. " +
            $"Actual: {_currentDataSet.Targets.IndexOf(_currentDataSet.Targets.Max())}");
        double errorSum = 0.0;
        for (int i = 0; i < outcomes.Count; i++)
        {
            errorSum += Math.Abs(_currentDataSet.Targets[i] - outcomes[i]);
        }
        Console.WriteLine($"Confidence: {Math.Round(Math.Abs((_currentDataSet.Targets[indexOutput] - outcomes[indexOutput]) / errorSum) * 100, 2) : 0.00}%\n");
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
        for (int i = 0; i < _config.HiddenLayers.Count; i++)
        {
            int previousInputSize = i == 0 ? _trainingData.First().Data.Count : _layers.Last().Count;
            CreateLayer(_config.HiddenLayers[i], previousInputSize);
        }
        CreateLayer(_trainingData.First().Targets.Count, _layers.Last().Count);
    }

    private void CreateLayer(int layerSize, int previousInputSize)
    {
        List<Neuron> layer = new();
        for (int i = 0; i < layerSize; i++)
        {
            Neuron neuron = new(GetRandomInRange(_config.BiasUpperLimit, _config.BiasLowerLimit), RandomizeWeights(previousInputSize));
            layer.Add(neuron);
        }
        _layers.Add(layer);
    }

    private void ForwardProp()
    {
        for (int i = 0; i < _layers.Count - 1; i++)
        {
            if(i == 0)
            {
                _layers[i].ForEach(x => x.Output = _config.HiddenFunction.Activation(x.ComputeOutput(_currentDataSet.Data)));
            }
            else
            {
                _layers[i].ForEach(x => x.Output = _config.HiddenFunction.Activation(x.ComputeOutput(_layers[i - 1].Select(y => y.Output).ToList())));
            }
        }
        List<double> finalOutput = _layers[^2].Select(y => y.Output).ToList();
        _layers[^1].ForEach(x => x.Output = x.ComputeOutput(finalOutput));
        _layers[^1].ForEach(x => x.Output = _config.OutcomeFunction.Activation(_layers[^1].Select(y => y.Output), x.Output));
    }

    private void BackwardProp()
    {
        for (int i = _layers.Count - 1; i > -1; i--)
        {
            for (int j = 0; j < _layers[i].Count; j++)
            {
                Neuron neuron = _layers[i][j];
                if (i == _layers.Count - 1)
                {
                    neuron.Gradient = CalculateOutcomeGradient(neuron.Output, _currentDataSet.Targets[j]);
                }
                else
                {
                    List<double> downstreamWeights = _layers[i + 1].Select(x => x.Weights[j]).ToList();
                    List<double> downnstreamGradients = _layers[i + 1].Select(x => x.Gradient).ToList();
                    neuron.Gradient = CalculateHiddenGradient(neuron.Output, downnstreamGradients, downstreamWeights);
                }
                for (int k = 0; k < neuron.Weights.Count; k++)
                {
                    List<double> upstreamOutputs = i == 0 ? _currentDataSet.Data : _layers[i - 1].Select(x => x.Output).ToList();
                    double weightDelta = CalculateDelta(upstreamOutputs[k], neuron.Gradient);
                    var move = weightDelta + CalculateMomentum(neuron.PreviousWeightDeltas[k]);
                    neuron.Weights[k] += move;
                    neuron.PreviousWeightDeltas[k] = weightDelta;
                }
                double biasDelta = CalculateDelta(1, neuron.Gradient);
                neuron.Bias += biasDelta + CalculateMomentum(neuron.PreviousBiasDelta);
                neuron.PreviousBiasDelta = biasDelta;
            }
        }
    }

    private double CalculateDelta(double previousOutput, double gradient)
    {
        return _config.DeltaCoefficient * previousOutput * gradient;
    }

    private double CalculateMomentum(double previousDelta)
    {
        return _config.MomentumCoefficient * previousDelta;
    }

    private double CalculateOutcomeGradient(double outcome, double desired)
    {
        return _config.OutcomeFunction.Derivative(outcome) * (desired - outcome);
    }

    private double CalculateHiddenGradient(double outcome, List<double> downstreamGradients, List<double> downstreamWeights)
    {
        double sum = 0;
        for (int i = 0; i < downstreamGradients.Count; i++)
        {
            sum += downstreamGradients[i] * downstreamWeights[i];
        }
        return _config.HiddenFunction.Derivative(outcome) * sum;
    }

    private List<double> RandomizeWeights(int layerSize)
    {  
        List<double> weights = new();
        for (int i = 0; i < layerSize; i++)
        {
            weights.Add(GetRandomInRange(_config.WeightUpperLimit, _config.WeightLowerLimit));
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
        _testingData = data.Take(quarterSetIndex).OrderBy(_ => _rnd.Next()).ToList();
        _trainingData = data.Skip(quarterSetIndex).OrderBy(_ => _rnd.Next()).ToList();
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