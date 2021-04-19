using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Numpy;
using PlotNET;
using Python.Runtime;

namespace Winaimp
{
    class Program
    {
        // the image dimensions and number of color channels
        static int imageHeight = 275;
        static int imageWidth = 348;
        static int latentDim = 100;
        static async Task Main(string[] args)
        {

            //TODO: check for non-square images
            imageWidth = 256;
            imageHeight = 256;

            var generator = DefineGenerator(latentDim);
            var discriminator = DefineDiscriminator();
            var gan = DefineGan(generator, discriminator);

            var count = 40000;
            var allfiles = Directory.GetFiles(@"input\").ToList();

            allfiles.Shuffle();

            var files = allfiles.Take(count);

            var allimgs = new List<NDarray>();

            foreach (var f in files)
            {
                var single = Keras.PreProcessing.Image.ImageUtil.LoadImg(f.Replace("//", "/"), "rgb", new Shape(new int[] { imageWidth, imageHeight }));
                var imgarray = Keras.PreProcessing.Image.ImageUtil.ImageToArray(single);

                var x = imgarray.astype(np.float32);
                NDarray xarry = (x - 127.5) / 127.5;
                allimgs.Add(xarry);
            }
            var data = np.array(allimgs.ToArray());
            allimgs = null;

            Train(generator, discriminator, gan, data, latentDim, 100, 16);
        }

        static void SummarizePerformance(int epoch, Sequential generator, Sequential discriminator, NDarray dataset, int latentDim, int sampleCount = 50)
        {
            var real = GenerateRealSamples(dataset, sampleCount);
            var realAcc = discriminator.Evaluate(real.Item1, real.Item2, verbose: 0);

            var fake = GenerateFakeGeneratorSamples(generator, latentDim, sampleCount);
            var fakeAcc = discriminator.Evaluate(fake.Item1, fake.Item2, verbose: 0);

            Console.WriteLine("Accuracy real: \t " + realAcc.Last() * 100 + "% \t fake:" + fakeAcc.Last() * 100 + "%");
            var fakes = fake.Item1;
            fakes = (fakes + 1) / 2;

            for (int i = 0; i < sampleCount; i++)
            {
                SaveArrayAsImage(fakes[i], "output/gantest_" + epoch + "_" + i + ".png");
            }

            generator.Save("output/generator" + epoch + ".h5");
        }

        static void Train(Sequential generator, Sequential discriminator, Sequential gan, NDarray dataset, int latentDim, int epochs = 200, int batchSize = 128)
        {
            var batchPerEpoch = dataset.shape[0] / batchSize;
            var halfBatch = batchSize / 2;

            for (var i = 0; i < epochs; i++)
            {
                for (var j = 0; j < batchPerEpoch; j++)
                {
                    var real = GenerateRealSamples(dataset, halfBatch);
                    var dLoss1 = discriminator.TrainOnBatch(real.Item1, real.Item2);
                    var fake = GenerateFakeGeneratorSamples(generator, latentDim, halfBatch);
                    var dLoss2 = discriminator.TrainOnBatch(fake.Item1, fake.Item2);

                    var xGan = GenerateLatentPoints(latentDim, batchSize);
                    var yGan = np.ones(new int[] { batchSize, 1 });
                    var gLoss = gan.TrainOnBatch(xGan, yGan);

                    Console.WriteLine("> EPOCH: " + i + " \t" + j + "/" + batchPerEpoch + " \td1=" + dLoss1.First() + " \td2=" + dLoss2.First() + " \tg=" + gLoss.Last() + "");

                }
                if (i % 10 == 0 || i < 10)
                {
                    SummarizePerformance(i, generator, discriminator, dataset, latentDim);
                }
            }
        }

        static void TrainGan(Sequential ganModel, int latentDim, int epochs, int batchSize)
        {
            for (int i = 0; i < epochs; i++)
            {
                var xGan = GenerateLatentPoints(latentDim, batchSize);
                var yGan = np.ones(new int[] { batchSize, 1 });
                ganModel.TrainOnBatch(xGan, yGan);
            }
        }

        static Sequential DefineGan(BaseModel generator, BaseModel discriminator)
        {
            discriminator.SetTrainable(false);

            var model = new Sequential();
            model.Add(generator);
            model.Add(discriminator);

            var opt = new Adam(0.0002f, 0.5f);
            model.Compile(loss: "binary_crossentropy", optimizer: opt);
            return model;
        }

        static Sequential DefineGenerator(int latentDim)
        {
            var model = new Sequential();
            int nodeCount = 256 * 8 * 8;
            model.Add(new Dense(nodeCount, input_dim: latentDim));
            model.Add(new LeakyReLU(0.2f));
            model.Add(new Reshape(new Shape(new int[] { 8, 8, 256 })));

            //16
            model.Add(new Conv2DTranspose(128, new Tuple<int, int>(4, 4), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            //32
            model.Add(new Conv2DTranspose(128, new Tuple<int, int>(4, 4), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            //64
            model.Add(new Conv2DTranspose(128, new Tuple<int, int>(4, 4), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            //128
            model.Add(new Conv2DTranspose(128, new Tuple<int, int>(4, 4), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            //256
            model.Add(new Conv2DTranspose(128, new Tuple<int, int>(4, 4), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Conv2D(3, new Tuple<int, int>(3, 3), activation: "tanh", padding: "same"));
            return model;
        }

        static Sequential DefineDiscriminator()
        {
            var model = new Sequential();
            model.Add(new Conv2D(64, new Tuple<int, int>(3, 3), padding: "same", input_shape: (imageWidth, imageHeight, 3)));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Conv2D(128, new Tuple<int, int>(3, 3), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Conv2D(128, new Tuple<int, int>(3, 3), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Conv2D(256, new Tuple<int, int>(3, 3), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Conv2D(256, new Tuple<int, int>(3, 3), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Conv2D(256, new Tuple<int, int>(3, 3), strides: new Tuple<int, int>(2, 2), padding: "same"));
            model.Add(new LeakyReLU(0.2f));

            model.Add(new Flatten());
            model.Add(new Dropout(0.4f));
            model.Add(new Dense(1, activation: "sigmoid"));

            var opt = new Adam(0.0002f, 0.5f);
            model.Compile(opt, loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            return model;

        }

        static NDarray GenerateLatentPoints(int latentDim, int sampleCount)
        {
            var xInput = np.random.randn(new int[] { latentDim * sampleCount });
            xInput = xInput.reshape(new int[] { sampleCount, latentDim });
            return xInput;
        }

        static (NDarray, NDarray) GenerateRealSamples(NDarray dataset, int sampleCount)
        {
            var ix = np.random.randint(0, dataset.shape[0], new int[] { sampleCount });
            var x = dataset[ix];
            var y = np.ones(new int[] { sampleCount, 1 });
            return (x, y);
        }

        static (NDarray, NDarray) GenerateFakeGeneratorSamples(Sequential generatorModel, int latentDim, int sampleCount)
        {
            var xInput = GenerateLatentPoints(latentDim, sampleCount);
            var x = generatorModel.Predict(xInput);
            var y = np.zeros(new int[] { sampleCount, 1 });
            return (x, y);
        }

        static (NDarray, NDarray) GenerateFakeSamples(int sampleCount)
        {
            var x = np.random.rand(imageWidth * imageHeight * 3 * sampleCount);
            x = -1 + x * 2;
            x = x.reshape(new int[] { sampleCount, imageWidth, imageHeight, 3 });
            var y = np.zeros(new int[] { sampleCount, 1 });
            return (x, y);
        }

        static void TrainDiscriminator(Sequential model, NDarray dataset, int iterations = 20, int batchSize = 128)
        {
            var halfBatch = batchSize / 2;

            for (int i = 0; i < iterations; i++)
            {
                var real = GenerateRealSamples(dataset, halfBatch);
                var realAcc = model.TrainOnBatch(real.Item1, real.Item2);
                var fake = GenerateFakeSamples(halfBatch);
                var fakeAcc = model.TrainOnBatch(fake.Item1, fake.Item2);
                Console.WriteLine(">" + i + "\t real=" + realAcc.Last() * 100 + "%\t fake=" + fakeAcc.Last() * 100 + "%");
            }
        }

        static void SaveArrayAsImage(NDarray array, string filename)
        {
            var b = Keras.PreProcessing.Image.ImageUtil.ArrayToImg(array);
            if (b is PyObject p)
            {
                p.InvokeMethod("save", new PyString[] { new PyString(filename) });
            }
        }
    }

    public static class Ext
    {
        public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> enumerable)
        {
            var r = new Random();
            return enumerable.OrderBy(x => r.Next()).ToList();
        }

        public static void SetTrainable(this BaseModel model, bool trainable)
        {
            model.ToPython().SetAttr("trainable", new PyInt(trainable ? 1 : 0));
        }

        public static BaseLayer[] Layers(this BaseModel model)
        {
            var lstLayers = new List<BaseLayer>();
            var layers = model.ToPython().GetAttr("layers");
            foreach (var layer in layers)
                lstLayers.Add(new BaseLayer(layer as PyObject));
            return lstLayers.ToArray();
        }

        public static BaseModel Add(this BaseModel model, BaseModel otherModel)
        {
            var b = model.ToPython();
            if (b is PyObject p)
            {
                p = p.InvokeMethod("add", new PyObject[] { otherModel.ToPython() });
                return p.As<BaseModel>();
            }
            return null;
        }

        private static NDarray ToNDarray(List<List<float>> batchList)
        {
            var inputArray = new float[batchList.Count(), batchList[0].Count()];

            for (int i = 0; i < batchList.Count(); i++)
            {
                for (int j = 0; j < batchList[0].Count(); j++)
                {
                    inputArray[i, j] = batchList[i][j];
                }
            }

            return np.array(inputArray);
        }
    }
}
