/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann
 */
#ifndef __UTILS_H__
#define __UTILS_H__

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

#endif //__UTILS_H__
