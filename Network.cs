using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace Suffer
{
    class Network
    {
        public int Layers { get; set; }
        public Neuron[][] Neurons { get; set; }
        public double[][][] Weights { get; set; }
        public bool useSIMD = true;
        public Network(int[] numbersOfNeurons)
        {
            Layers = numbersOfNeurons.Length;
            Neurons = new Neuron[Layers][];
            Weights = new double[Layers - 1][][];

            int i = 0;
            foreach (var numNeurons in numbersOfNeurons)
            {
                InitializeNeurons(i, numNeurons);

                if (i < Layers - 1)
                {
                    Weights[i] = new double[numNeurons][];
                    for (int j = 0; j < numNeurons; j++)
                    {
                        Weights[i][j] = new double[numbersOfNeurons[i + 1]];
                    }
                }

                i++;
            }
        }

        private void InitializeNeurons(int layerIndex, int numNeurons)
        {
            Neurons[layerIndex] = new Neuron[numNeurons];
            for (int j = 0; j < numNeurons; j++)
            {
                Neurons[layerIndex][j] = new Neuron();
            }
        }

        public void SetRandomWeights()
        {
            Random rnd = new Random();
            for (int i = 0; i < Layers - 1; i++)
            {
                int currentLayerSize = Neurons[i].Length;
                int nextLayerSize = Neurons[i + 1].Length;
                Weights[i] = new double[currentLayerSize][];

                for (int j = 0; j < currentLayerSize; j++)
                {
                    Weights[i][j] = new double[nextLayerSize];
                    for (int k = 0; k < nextLayerSize; k++)
                    {
                        Weights[i][j][k] = rnd.NextDouble() * 2 - 1; // Random double between -1 and 1
                    }
                }
            }
        }

        public void SetInput(double[] values)
        {
            if (values.Length != Neurons[0].Length)
            {
                throw new ArgumentException("The number of input values does not match the number of input neurons.");
            }

            int i = 0;
            foreach (var neuron in Neurons[0])
            {
                neuron.Value = values[i++];
            }
        }







       
        public void ForwardFeed()
        {
            for (int k = 1; k < Layers; k++)
            {
                for (int i = 0; i < Neurons[k].Length; i++)
                {
                    Neurons[k][i].Value = 0;
                    for (int j = 0; j < Neurons[k - 1].Length; j++)
                    {
                        Neurons[k][i].Value += Neurons[k - 1][j].Value * Weights[k - 1][j][i];
                    }
                    Neurons[k][i].Activation();
                    Neurons[k][i].Index = i; // set the index of the neuron
                }
            }
        }

    

        
        public void BackPropogation(int rightResult, double learningRate)
        {
            // Output layer errors
            for (int i = 0; i < Neurons[Layers - 1].Length; i++)
            {
                if (rightResult == i)
                {
                    Neurons[Layers - 1][i].Error = Neurons[Layers - 1][i].Value - 1.0;
                }
                else
                {
                    Neurons[Layers - 1][i].Error = Neurons[Layers - 1][i].Value;
                }
            }

            // Hidden layer errors
            for (int i = Layers - 2; i >= 0; i--)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    double error = 0.0;
                    for (int k = 0; k < Neurons[i + 1].Length; k++)
                    {
                        error += Neurons[i + 1][k].Error * Weights[i][j][k];
                    }
                    Neurons[i][j].Error = error * SigmoidDerivative(Neurons[i][j].Value);
                }
            }

            // Update weights
            for (int i = 0; i < Layers - 1; i++)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    for (int k = 0; k < Neurons[i + 1].Length; k++)
                    {
                        Weights[i][j][k] -= learningRate * Neurons[i + 1][k].Error * Neurons[i][j].Value;
                    }
                }
            }
        }
        

      




        public int GetMaxNeuronIndex(int layerNumber)
        {
            double maxValue = double.MinValue;
            int maxIndex = -1;

            for (int i = 0; i < Neurons[layerNumber].Length; i++)
            {
                if (Neurons[layerNumber][i].Value > maxValue)
                {
                    maxValue = Neurons[layerNumber][i].Value;
                    maxIndex = i;
                }
            }

            return maxIndex;
        }


        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public static double SigmoidDerivative(double x)
        {
            if (Math.Abs(x - 1) < 1e-9 || Math.Abs(x) < 1e-9) return 0.0;
            double result = x * (1.0 - x);
            return result;
        }

        public void SaveWeightsToFile(string path)
        {
            string text = "";
            for (int i = 0; i < Layers - 1; i++)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    for (int k = 0; k < Neurons[i + 1].Length; k++)
                    {
                        text += Weights[i][j][k] + " ";
                    }
                }
            }
            using (FileStream file = new FileStream(path, FileMode.Create))
            {
                using (StreamWriter stream = new StreamWriter(file))
                {
                    stream.Write(text);
                }
            }
        }

        public void LoadWeightsFromFile(string path)
        {
            string text = File.ReadAllText(path);
            string[] textWeights = text.Split(' ');
            int c = 0;
            for (int i = 0; i < Layers - 1; i++)
            {
                for (int j = 0; j < Neurons[i].Length; j++)
                {
                    for (int k = 0; k < Neurons[i + 1].Length; k++)
                    {
                        Weights[i][j][k] = Convert.ToDouble(textWeights[c]);
                        c++;
                    }
                }
            }
        }
    }
}
