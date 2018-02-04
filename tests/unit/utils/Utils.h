/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann
 */
#ifndef __UTILS_H__
#define __UTILS_H__

#include <shogun/lib/SGMatrix.h>
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

#endif //__UTILS_H__
