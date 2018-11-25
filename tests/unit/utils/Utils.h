/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann
 */
#ifndef __UTILS_H__
#define __UTILS_H__

#include "shogun/mathematics/Math.h"
#include "shogun/neuralnets/NeuralInputLayer.h"
#include "shogun/neuralnets/NeuralLinearLayer.h"
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

/** Generate file name for serialization test
 *
 * @param file_name template of file name
 */
void generate_temp_filename(char* file_name);

/** Generate toy weather data
 *
 * @param data feature matrix to be set, shape = [n_features, n_samples]
 * @param labels labels vector to be set, shape = [n_samples]
 */
void generate_toy_data_weather(
    SGMatrix<float64_t>& data, SGVector<float64_t>& labels,
    bool load_train_data = true);

template <typename T = char>
SGStringList<T> generateRandomStringData(
    index_t num_strings = 10, index_t max_string_length = 10,
    index_t min_string_length = 10)
{
	SGStringList<T> strings(num_strings, max_string_length);

	for (index_t i = 0; i < num_strings; ++i)
	{
		index_t len =
		    std::rand() % (max_string_length - min_string_length + 1) +
		    min_string_length;
		SGString<T> current(len);
		/* fill with random uppercase letters (ASCII) */
		for (index_t j = 0; j < len; ++j)
		{
			current.string[j] = (T)(std::rand() % ('Z' - 'A' + 1) + 'A');
			T* string = SG_MALLOC(T, 2);
			string[0] = current.string[j];
			string[1] = '\0';
			SG_FREE(string);
		}

		strings.strings[i] = current;
	}
	return strings;
}

struct NeuralLayerTestUtil
{
	template <typename T>
	static SGMatrix<T> create_rand_sgmat(
	    int32_t num_rows, int32_t num_cols, T lower_bound, T upper_bound)
	{
		auto data_batch = SGMatrix<T>(num_rows, num_cols);
		for (int32_t i = 0; i < data_batch.num_rows * data_batch.num_cols; i++)
		{
			data_batch[i] = CMath::random(lower_bound, upper_bound);
		}
		return data_batch;
	}

	template <typename T>
	static auto create_rand_input_layer(
	    int32_t num_features, int32_t num_samples, T lower_bound, T upper_bound)
	{
		auto data_batch = create_rand_sgmat(
		    num_features, num_samples, lower_bound, upper_bound);
		// Can't use unique_ptr here due to "unref_all"
		auto input_layer = new CNeuralInputLayer(data_batch.num_rows);
		input_layer->set_batch_size(data_batch.num_cols);
		input_layer->compute_activations(data_batch);
		return std::make_tuple(data_batch, input_layer);
	}

	static auto init_neural_linear_layer(
	    CNeuralLinearLayer* layer, CDynamicObjectArray* layers,
	    SGVector<int32_t> input_indices, int32_t batch_size, double sigma)
	{
		layer->initialize_neural_layer(layers, input_indices);
		SGVector<float64_t> params(layer->get_num_parameters());
		SGVector<bool> param_regularizable(layer->get_num_parameters());
		layer->initialize_parameters(params, param_regularizable, sigma);
		layer->set_batch_size(batch_size);
		layer->compute_activations(params, layers);
		return params;
	}
};
#endif //__UTILS_H__
