
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Media;
using System.Text.RegularExpressions;


namespace Suffer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        
        bool painting = false;
        System.Windows.Media.Brush brushcolor = System.Windows.Media.Brushes.White;
        const int SIZE = 28;
        Network network = new Network(new int[] { 784, 250, 100, 10 });
        double[] input = new double[SIZE * SIZE];
        double[,] colors = new double[SIZE, SIZE];
        string path = "C:\\Users\\its_m\\source\\repos\\Suffer\\bin\\Debug\\net6.0-windows\\weights.txt";
        string Path = "C:\\Users\\its_m\\source\\repos\\Suffer\\bin\\Debug\\net6.0-windows";
        

        public MainWindow()
        {
            network.LoadWeightsFromFile("weights_ideal.txt");
            InitializeComponent();
        }


        private void paintCircle(System.Windows.Media.Brush color,   System.Windows.Point position)
        {
            Ellipse ellipse = new Ellipse();
            ellipse.Fill = color;
            ellipse.Width = 20;
            ellipse.Height = 20;
            Canvas.SetTop(ellipse, position.Y);
            Canvas.SetLeft(ellipse, position.X);
            inkCanvas.Children.Add(ellipse);

        }
        private void Button_Click(object sender, RoutedEventArgs e)
        {
            inkCanvas.Children.Clear();
            Prediction.Text = "";
        }

        private void PredictionButton_Click(object sender, RoutedEventArgs e)
        {

            // get the bounds of the canvas
            // get the dimensions of the ink control
            double[] input = Magic();
            network.SetInput(input);
            network.ForwardFeed();

            Prediction.Text = network.GetMaxNeuronIndex(network.Layers - 1) + "";

        }

        private double[] Magic()
        {
            int margin = (int)this.inkCanvas.Margin.Left;
            int width = (int)this.inkCanvas.ActualWidth - margin;
            int height = (int)this.inkCanvas.ActualHeight - margin;

            // render ink to bitmap
            RenderTargetBitmap rtb = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Default);
            rtb.Render(inkCanvas);

            // resize the bitmap to 28x28
            WriteableBitmap wb = new WriteableBitmap(rtb);
            TransformedBitmap tb = new TransformedBitmap(wb, new ScaleTransform(28.0 / wb.PixelWidth, 28.0 / wb.PixelHeight, 0.0, 0.0));

            // create a byte array to hold the pixel data
            int stride = tb.PixelWidth * 4;
            byte[] pixelData = new byte[tb.PixelHeight * stride];
            tb.CopyPixels(pixelData, stride, 0);

            // create a list to hold the pixel values
            List<double> pixels = new List<double>();

            // loop over the pixels in the byte array and get their values
            for (int y = 0; y < tb.PixelHeight; y++)
            {
                for (int x = 0; x < tb.PixelWidth; x++)
                {
                    int index = y * stride + 4 * x;
                    byte blue = pixelData[index];
                    byte green = pixelData[index + 1];
                    byte red = pixelData[index + 2];
                    byte alpha = pixelData[index + 3];

                    // convert the pixel values to doubles and add them to the list
                    double pixelValue = blue / 255.0;
                    pixels.Add(pixelValue);
                }
            }

            // convert the list to an array and pass it as input to the neural network
            double[] input = pixels.ToArray();
            return input;
        }

      

        private void LearningButton_Click(object sender, RoutedEventArgs e)
        {
            double prevcor = 0.0;
            double rightRate = 0, errRate;
            double LR = 0.01;
            double right, err;
            int result, expect;
            Network network = new Network(new int[] { 784, 32, 16, 10 });
            Random rnd = new Random();

            var outputs = new List<int>();
            var inputs = new List<double[]>();

            //network.SetRandomWeights();
            network.LoadWeightsFromFile("weights.txt");
            Console.WriteLine("Weights loaded");
            using (var sr = new StreamReader("mnist_test.csv"))
            {
                var header = sr.ReadLine();
                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    var values = row.Split(',').Select(v => Convert.ToDouble(v)).ToArray();
                    var output = (int)values.First();
                    var input = values.Skip(1).Select(v => v / 255.0).ToArray();
                    outputs.Add(output);
                    inputs.Add(input);
                }
            }
            int epoch = 0;
            right = 0;
            err = 0;
            do
            {
                epoch++;
                right = 0;
                err = 0;
                var batchsize = 100;
                for (int j = 0; j < batchsize; j++)
                {
                    var itemNum = rnd.Next(outputs.Count);
                    expect = outputs[itemNum];
                    network.SetInput(inputs[itemNum]);
                    network.ForwardFeed();
                    result = network.GetMaxNeuronIndex(network.Layers - 1);
                    if (result == expect)
                        right++;
                    else
                        err++;
                    network.BackPropogation(expect, LR);
                }
                prevcor = rightRate;
                rightRate = right / batchsize * 100;
                errRate = err / batchsize * 100;
                Console.WriteLine($"Epoch {epoch} Correct {rightRate}% Error {errRate}%");
            } while (rightRate < 97);

            network.SaveWeightsToFile("weights.txt");
            Console.WriteLine("Weights saved");
        }




        private void inkCanvas_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            painting = false;

        }

        private void inkCanvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            painting = true;
        }

        private void inkCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if(painting)
            {
                System.Windows.Point pos = e.GetPosition(inkCanvas);
                paintCircle(brushcolor, pos);
            }
        }

        private void Incorrect_Click(object sender, RoutedEventArgs e)
        {
                network.BackPropogation(Convert.ToInt32(Correcter.Text), 0.1);
                double[] input = Magic();
                network.SetInput(input);
                network.ForwardFeed();
                Prediction.Text = network.GetMaxNeuronIndex(network.Layers - 1) + "";
        }


        /*
        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            e.Handled = new Regex("[^[0-9]?]").IsMatch(e.Text);
        }
        */
        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            // Regular expression pattern to match valid numbers (integer or decimal)
            string pattern = @"^[-+]?\d*\.?\d*$";

            // Check if the input matches the pattern
            if (!Regex.IsMatch(e.Text, pattern))
            {
                // If not a valid number, mark the event as handled to prevent the input from being added to the TextBox
                e.Handled = true;
                return;
            }

            // Combine the existing text and the new input
            string combinedText = (sender as TextBox).Text.Remove((sender as TextBox).SelectionStart, (sender as TextBox).SelectionLength);
            combinedText = combinedText.Insert((sender as TextBox).SelectionStart, e.Text);

            // Parse the combined text to a double and check if it's between 0 and 9
            double number;
            if (!double.TryParse(combinedText, out number) || number < 0 || number > 9)
            {
                // If not between 0 and 9, mark the event as handled to prevent the input from being added to the TextBox
                e.Handled = true;
            }
        }



        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            //network.SaveWeightsToFile(path);
        }

        private void Window_Close(object sender, EventArgs e)
        {
            //network.SaveWeightsToFile(path);
        }

        private void Correcter_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(Correcter.Text))
            {
                Incorrect.IsEnabled = false;
            }
            else
            {
                Incorrect.IsEnabled = true;
            }
        }
    }
}
