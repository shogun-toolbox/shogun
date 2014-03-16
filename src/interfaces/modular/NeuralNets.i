/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) Khaled Nasr
 */

%newobject apply();
 
/* Remove C Prefix */
%rename(NeuralNetwork) CNeuralNetwork;
%rename(NeuralLayer) CNeuralLayer;
%rename(NeuralLinearLayer) CNeuralLinearLayer;
%rename(NeuralLogisticLayer) CNeuralLogisticLayer;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/neuralnets/NeuralNetwork.h>
%include <shogun/neuralnets/NeuralLayer.h>
%include <shogun/neuralnets/NeuralLinearLayer.h>
%include <shogun/neuralnets/NeuralLogisticLayer.h>
 
