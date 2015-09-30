/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation 
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its 
 * contributors may be used to endorse or promote products derived from this 
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * Written (W) 2014 Khaled Nasr
 */

#ifndef __NEURALNETWORKFILEREADER_H__
#define __NEURALNETWORKFILEREADER_H__

#include <shogun/lib/config.h>

#ifdef HAVE_JSON

#include <json.h>

#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
class CNeuralNetwork;
class CNeuralLayer;
class CDynamicObjectArray;

/** @brief Creates a CNeuralNetwork from a JSON file
 * 
 * The class creates and initializes a CNeuralNetwork using a JSON file. The 
 * file can be used to specify the structure of the network (types of layers, 
 * connections between layers, ..) and the training parameters.
 * 
 * The file must contain a JSON object with the name "layers". This describes 
 * the layers of the network. Each layer must have the following: 
 * 	- a "type" field that contains the name of the class to be created, without 
 * the C, i.e "NeuralLogisticLayer" for CNeuralLogisticLayer.
 * 	- a "num_neurons" field that contains the number of neurons
 * 	- For CNeuralInputLayer, a "start_index" field could be added to specify 
 * the start index
 * 	- For non-input layers, an "inputs" field must be specified, which is an 
 * array of strings (names of other layer) that specify which layers are 
 * connected to this layer as input.
 * 
 * The training parameters can also be specified (see example below). For a 
 * list of all possible parameters, see the public variable of CNeuralNetwork.
 * 
 * A field named "sigma" could be added to specify the standard deviation of 
 * the gaussian used to initialize the parameters.
 * 
 * The following example shows how to create a network with two input layers, 
 * one logistic hidden layer, one rectified-linear hidden layer and a softmax 
 * output layer. The logistic layer takes input from the first input layer. 
 * The rectified-linear layer takes input from the second input layer and the 
 * logistic layer. The softmax layer takes input from the logistic layer and the 
 * rectified-linear layer. The network uses dropout and is trained with gradient 
 * descent.
 * 
 * \code{.json}
	{
		"sigma": 0.01,
		
		"l2_coefficient": 0.001,
		
		"optimization_method": "NNOM_GRADIENT_DESCENT",
		
		"dropout_hidden": 0.5,
		
		"dropout_input": 0.2,
		
		"layers":
		{
			"input1":
			{
				"type": "NeuralInputLayer",
				"num_neurons": 6,
				"start_index": 0
			},
			"input2":
			{
				"type": "NeuralInputLayer",
				"num_neurons": 10,
				"start_index": 6
			},
			"logistic":
			{
				"type": "NeuralLogisticLayer",
				"num_neurons": 32,
				"inputs": ["input1"]
			},
			"rectified":
			{
				"type": "NeuralRectifiedLinearLayer",
				"num_neurons": 8,
				"inputs": ["logistic", "input2"]
			},
			"softmax":
			{
				"type": "NeuralSoftmaxLayer",
				"num_neurons": 4,
				"inputs": ["logistic", "rectified"]
			}
		}
	}
 * \endcode
 * 
 */
class CNeuralNetworkFileReader : public CSGObject
{
public:
	/** default constructor */
	CNeuralNetworkFileReader() { }
	
	virtual ~CNeuralNetworkFileReader() { }
	
	/** Reads a given JSON file
	 * 
	 * @param file_path path to the JSON file describing the network
	 *
	 * @return newly created and initialized neural network according to the 
	 * contents of the file 
	 */
	virtual CNeuralNetwork* read_file(const char* file_path);
	
	/** Reads a given JSON string
	 * 
	 * @param str string containing JSON code describing the network
	 *
	 * @return newly created and initialized neural network according to the 
	 * contents of the string 
	 */
	virtual CNeuralNetwork* read_string(const char* str);
	
	virtual const char* get_name() const { return "NeuralNetworkFileReader";}

protected:
	/** Parses a JSON object into a CNeuralNetwork */
	virtual CNeuralNetwork* parse_network(json_object* json_network);
	
	/** Parses a JSON object into a CDynamicObjectArray of CNeuralLayer objects 
	 */
	virtual CDynamicObjectArray* parse_layers(json_object* json_layers);
	
	/** Parses a JSON object into a CNeuralLayer */
	virtual CNeuralLayer* parse_layer(json_object* json_layer);
	
private:
	/** Attempts to find a layer with the given key in the given array of 
	 * layers. Return the index of the layer if found, -1 otherwise
	 */
	int32_t find_layer_index(json_object* json_layers, const char* layer_key);
	
	/** Returns true if the two given strings match completely, false otherwise 
	 */
	bool string_equal(const char* str1, const char* str2);
};

}
#endif /* HAVE_JSON  */
#endif /* __NEURALNETWORKFILEREADER_H__  */
