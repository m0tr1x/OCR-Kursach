﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Suffer
{
    class Neuron
    {
        public int Index { get; set; }
        public double Value { get; set; }
        public double Error { get; set; }

        public void Activation()
        {
            Value = Network.Sigmoid(Value);
        }
    }
}
