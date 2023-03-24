namespace NNFun;

internal static class MnistReader
{
    private const string TrainImages = "mnist/train-images.idx3-ubyte";
    private const string TrainLabels = "mnist/train-labels.idx1-ubyte";
    private const string TestImages = "mnist/t10k-images.idx3-ubyte";
    private const string TestLabels = "mnist/t10k-labels.idx1-ubyte";

    internal static List<DataSet> GetData()
    {
        List<DataSet> data = new();
        data.AddRange(ReadTestData());
        data.AddRange(ReadTrainingData());
        return data;
    }

    private static IEnumerable<DataSet> ReadTrainingData()
    {
        foreach (var item in Read(TrainImages, TrainLabels))
        {
            yield return item;
        }
    }

    private static IEnumerable<DataSet> ReadTestData()
    {
        foreach (var item in Read(TestImages, TestLabels))
        {
            yield return item;
        }
    }

    private static IEnumerable<DataSet> Read(string imagesPath, string labelsPath)
    {
        BinaryReader labels = new(new FileStream(labelsPath, FileMode.Open));
        BinaryReader images = new(new FileStream(imagesPath, FileMode.Open));

        int magicNumber = images.ReadBigInt32();
        int numberOfImages = images.ReadBigInt32();
        int width = images.ReadBigInt32();
        int height = images.ReadBigInt32();

        int magicLabel = labels.ReadBigInt32();
        int numberOfLabels = labels.ReadBigInt32();

        for (int i = 0; i < numberOfImages; i++)
        {
            byte[] bytes = images.ReadBytes(width * height);
            List<double> values = bytes.Select(x => (double)x).ToList();

            List<double> targets = Enumerable.Repeat(0.0, 10).ToList();
            var labelIndex = (int)labels.ReadByte();
            targets[labelIndex] = 1;

            yield return new DataSet(values.ToList(), targets);
        }
    }
}

public static class Extensions
{
    public static int ReadBigInt32(this BinaryReader br)
    {
        var bytes = br.ReadBytes(sizeof(int));
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }

    public static void ForEach<T>(this T[,] source, Action<int, int> action)
    {
        for (int w = 0; w < source.GetLength(0); w++)
        {
            for (int h = 0; h < source.GetLength(1); h++)
            {
                action(w, h);
            }
        }
    }
}