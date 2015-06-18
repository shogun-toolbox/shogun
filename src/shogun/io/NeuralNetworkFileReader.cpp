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

#include <shogun/lib/config.h>
#ifdef HAVE_JSON

#include <shogun/io/NeuralNetworkFileReader.h>
#include <shogun/neuralnets/NeuralNetwork.h>
#include <shogun/neuralnets/NeuralLayer.h>
#include <shogun/neuralnets/NeuralInputLayer.h>
#include <shogun/neuralnets/NeuralLinearLayer.h>
#include <shogun/neuralnets/NeuralLogisticLayer.h>
#include <shogun/neuralnets/NeuralSoftmaxLayer.h>
#include <shogun/neuralnets/NeuralRectifiedLinearLayer.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

CNeuralNetwork* CNeuralNetworkFileReader::read_file(const char* file_path)
{
	json_object* json_network = json_object_from_file(file_path);

	if (is_error(json_network))
	{
		SG_ERROR("Error while opening file: %s!\n", file_path);
		return NULL;
	}

	CNeuralNetwork* network = parse_network(json_network);

	json_object_put(json_network);

	return network;
}

CNeuralNetwork* CNeuralNetworkFileReader::read_string(const char* str)
{
	json_object* json_network = json_tokener_parse(str);

	if (is_error(json_network))
	{
		SG_ERROR("Error while parsing the given string\n");
		return NULL;
	}

	CNeuralNetwork* network = parse_network(json_network);

	json_object_put(json_network);

	return network;
}

CNeuralNetwork* CNeuralNetworkFileReader::parse_network(json_object* json_network)
{
	CNeuralNetwork* network = new CNeuralNetwork;

	// find the layers
	json_object_iter iter;
	json_object* json_layers = NULL;
	json_object_object_foreachC(json_network, iter)
	{
		if (string_equal(iter.key, "layers"))
			json_layers = iter.val;
	}

	if (json_layers)
		network->set_layers(parse_layers(iter.val));
	else
		SG_ERROR("No layers found in file\n");

	// set the connections
	json_object_iter layers_iter;
	json_object_object_foreachC(json_layers, layers_iter)
	{
		json_object_iter layer_iter;
		json_object_object_foreachC(layers_iter.val, layer_iter)
		{
			if (string_equal(layer_iter.key, "inputs"))
			{
				int32_t len = json_object_array_length(layer_iter.val);

				for (int32_t i=0; i<len; i++)
				{
					const char* input_key = json_object_get_string(
						json_object_array_get_idx(layer_iter.val, i));

					int32_t from = find_layer_index(json_layers, input_key);
					int32_t to = find_layer_index(json_layers, layers_iter.key);

					if (from == -1)
						SG_ERROR("Invalid layer identifier (%s) in layer (%s)\n",
							input_key, layers_iter.key);

					network->connect(from, to);
				}
			}
		}
	}

	// set the training parameters
	float sigma = 0.01;
	json_object_object_foreachC(json_network, iter)
	{
		if (string_equal(iter.key, "sigma"))
			sigma = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "optimization_method"))
		{
			const char* method = json_object_get_string(iter.val);
			if (string_equal(method, "NNOM_LBFGS"))
				network->optimization_method = NNOM_LBFGS;
			else if (string_equal(method, "NNOM_GRADIENT_DESCENT"))
				network->optimization_method = NNOM_GRADIENT_DESCENT;
			else
				SG_ERROR("Invalid optimization method (%s)\n", method);
		}
		else if (string_equal(iter.key, "l2_coefficient"))
			network->l2_coefficient = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "l1_coefficient"))
			network->l1_coefficient = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "dropout_hidden"))
			network->dropout_hidden = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "dropout_input"))
			network->dropout_input = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "max_norm"))
			network->max_norm = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "epsilon"))
			network->epsilon = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "max_num_epochs"))
			network->max_num_epochs = json_object_get_int(iter.val);
		else if (string_equal(iter.key, "gd_mini_batch_size"))
			network->gd_mini_batch_size = json_object_get_int(iter.val);
		else if (string_equal(iter.key, "gd_learning_rate"))
			network->gd_learning_rate = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "gd_learning_rate_decay"))
			network->gd_learning_rate_decay = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "gd_momentum"))
			network->gd_momentum = json_object_get_double(iter.val);
		else if (string_equal(iter.key, "gd_error_damping_coeff"))
			network->gd_error_damping_coeff = json_object_get_double(iter.val);

		else if (!string_equal(iter.key, "layers"))
			SG_ERROR("Invalid parameter (%s)\n", iter.key);
	}

	network->initialize_neural_net(sigma);

	return network;
}

CDynamicObjectArray* CNeuralNetworkFileReader::parse_layers(
		json_object* json_layers)
{
	CDynamicObjectArray* layers = new CDynamicObjectArray();

	json_object_iter iter;
	json_object_object_foreachC(json_layers, iter)
	{
		layers->append_element(parse_layer(iter.val));
	}

	return layers;
}

CNeuralLayer* CNeuralNetworkFileReader::parse_layer(json_object* json_layer)
{
	json_object_iter iter;

	CNeuralLayer* layer = NULL;
	const char* type = NULL;

	// find the layer type and create a appropriate instance
	json_object_object_foreachC(json_layer, iter)
	{
		if (string_equal(iter.key, "type"))
		{
			type = json_object_get_string(iter.val);

			if (string_equal(type, "NeuralInputLayer"))
				layer = new CNeuralInputLayer();
			else if (string_equal(type, "NeuralLinearLayer"))
				layer = new CNeuralLinearLayer();
			else if (string_equal(type, "NeuralLogisticLayer"))
				layer = new CNeuralLogisticLayer();
			else if (string_equal(type, "NeuralSoftmaxLayer"))
				layer = new CNeuralSoftmaxLayer();
			else if (string_equal(type, "NeuralRectifiedLinearLayer"))
				layer = new CNeuralRectifiedLinearLayer();
			else
				SG_ERROR("Unknown layer type: %s", type);
		}
	}

	// fill in the fields
	json_object_object_foreachC(json_layer, iter)
	{
		if(string_equal(iter.key, "num_neurons"))
		{
			layer->set_num_neurons(json_object_get_int(iter.val));
		}
		else if(string_equal(type,"NeuralInputLayer") &&
			string_equal(iter.key, "start_index"))
		{
			((CNeuralInputLayer*)layer)->set_start_index(
				json_object_get_int(iter.val));
		}
	}

	return layer;
}

int32_t CNeuralNetworkFileReader::find_layer_index(json_object* json_layers,
		const char* layer_key)
{
	int32_t index = 0;

	json_object_iter iter;
	json_object_object_foreachC(json_layers, iter)
	{
		if (string_equal(iter.key, layer_key))
			return index;
		else
			index++;
	}

	return -1;
}

bool CNeuralNetworkFileReader::string_equal(const char* str1, const char* str2)
{
	return (strcmp(str1, str2) == 0);
}

#endif
