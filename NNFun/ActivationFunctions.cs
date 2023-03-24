namespace NNFun;

internal interface IFunction
{
    double Derivative(double output);
}
internal interface IHiddenFunction : IFunction
{
    double Activation(double output);
}

internal interface IOutcomeFunction : IFunction
{
    double Activation(IEnumerable<double> outputs, double currentOutput);
}

internal class ReLU : IHiddenFunction
{
    public double Activation(double output)
    {
        return output > 0 ? output : 0;
    }

    public double Derivative(double output)
    {
        return output > 0 ? 1 : 0;
    }
}

internal class Sigmoid : IHiddenFunction
{
    public double Activation(double output)
    {
        return 1 / (1 + Math.Exp(-output));
    }

    public double Derivative(double output)
    {
        return output * (1 - output);
    }
}

internal class SoftMax : IOutcomeFunction
{
    public double Activation(IEnumerable<double> outputs, double currentOutput)
    {
        IEnumerable<double> exp = outputs.Select(Math.Exp);
        double expSum = exp.Sum();
        return Math.Exp(currentOutput) / expSum;
    }

    public double Derivative(double output)
    {
        return (1 - output) * output;
    }
}
