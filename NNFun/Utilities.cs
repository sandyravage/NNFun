namespace NNFun;

internal static class Utilities
{
    public static List<List<double>> MultiplyMatrices(List<List<double>> a, List<List<double>> b)
    {
        if(a.FirstOrDefault()?.Count != b.Count)
        {
            throw new MatrixMultiplicationException("Matrices are not able to be multiplied");
        }

        List<List<double>> result = new();

        for (int i = 0; i < a.Count; i++)
        {
            List<double> row = new();   
            for (int j = 0; j < b[i].Count; j++)
            {
                double rowValue = 0;
                for (int k = 0; k < a[i].Count; k++)
                {
                    rowValue += a[i][k] * b[k][j];
                }
                row.Add(rowValue);
            }
            result.Add(row);
        }

        return result;
    }
}

internal class MatrixMultiplicationException : Exception
{
    public MatrixMultiplicationException(string message) : base(message)
    {

    }
};
