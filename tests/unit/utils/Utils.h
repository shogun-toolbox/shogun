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

SGStringList<char> generateRandomStringData(
    index_t num_strings = 10, index_t max_string_length = 20,
    index_t min_string_length = 10);

#endif //__UTILS_H__
