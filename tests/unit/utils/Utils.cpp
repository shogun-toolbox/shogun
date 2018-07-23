/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann
 */
#include "Utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <string>

#ifdef _MSC_VER
#include <io.h>
#endif

void generate_temp_filename(char* file_name)
{
#ifdef _WIN32
	int err = _mktemp_s(file_name, strlen(file_name) + 1);
	ASSERT(err == 0);
#else
	int fd = mkstemp(file_name);
	ASSERT(fd != -1);
	int retval = close(fd);
	ASSERT(retval != -1);
#endif
}

SGStringList<char> generateRandomStringData(
    index_t num_strings, index_t max_string_length, index_t min_string_length)
{
	SGStringList<char> strings(num_strings, max_string_length);

	for (index_t i = 0; i < num_strings; ++i)
	{
		index_t len = CMath::random(min_string_length, max_string_length);
		SGString<char> current(len);
		/* fill with random uppercase letters (ASCII) */
		for (index_t j = 0; j < len; ++j)
		{
			current.string[j] = (char)CMath::random('A', 'Z');
			char* string = SG_MALLOC(char, 2);
			string[0] = current.string[j];
			string[1] = '\0';
			SG_FREE(string);
		}

		strings.strings[i] = current;
	}
	return strings;
}

void generate_toy_data_weather(
    SGMatrix<float64_t>& data, SGVector<float64_t>& labels,
    bool load_train_data)
{
	const double sunny = 1.;
	const double overcast = 2.;
	const double rain = 3.;

	const double hot = 1.;
	const double mild = 2.;
	const double cool = 3.;

	const double high = 1.;
	const double normal = 2.;

	const double weak = 1.;
	const double strong = 2.;

	// vector = [Outlook Temperature Humidity Wind]
	if (load_train_data)
	{
		data(0, 0) = sunny;
		data(1, 0) = hot;
		data(2, 0) = high;
		data(3, 0) = weak;

		data(0, 1) = sunny;
		data(1, 1) = hot;
		data(2, 1) = high;
		data(3, 1) = strong;

		data(0, 2) = overcast;
		data(1, 2) = hot;
		data(2, 2) = high;
		data(3, 2) = weak;

		data(0, 3) = rain;
		data(1, 3) = mild;
		data(2, 3) = high;
		data(3, 3) = weak;

		data(0, 4) = rain;
		data(1, 4) = cool;
		data(2, 4) = normal;
		data(3, 4) = weak;

		data(0, 5) = rain;
		data(1, 5) = cool;
		data(2, 5) = normal;
		data(3, 5) = strong;

		data(0, 6) = overcast;
		data(1, 6) = cool;
		data(2, 6) = normal;
		data(3, 6) = strong;

		data(0, 7) = sunny;
		data(1, 7) = mild;
		data(2, 7) = high;
		data(3, 7) = weak;

		data(0, 8) = sunny;
		data(1, 8) = cool;
		data(2, 8) = normal;
		data(3, 8) = weak;

		data(0, 9) = rain;
		data(1, 9) = mild;
		data(2, 9) = normal;
		data(3, 9) = weak;

		data(0, 10) = sunny;
		data(1, 10) = mild;
		data(2, 10) = normal;
		data(3, 10) = strong;

		data(0, 11) = overcast;
		data(1, 11) = mild;
		data(2, 11) = high;
		data(3, 11) = strong;

		data(0, 12) = overcast;
		data(1, 12) = hot;
		data(2, 12) = normal;
		data(3, 12) = weak;

		data(0, 13) = rain;
		data(1, 13) = mild;
		data(2, 13) = high;
		data(3, 13) = strong;

		labels[0] = 0.0;
		labels[1] = 0.0;
		labels[2] = 1.0;
		labels[3] = 1.0;
		labels[4] = 1.0;
		labels[5] = 0.0;
		labels[6] = 1.0;
		labels[7] = 0.0;
		labels[8] = 1.0;
		labels[9] = 1.0;
		labels[10] = 1.0;
		labels[11] = 1.0;
		labels[12] = 1.0;
		labels[13] = 0.0;
	}
	else
	{
		data(0, 0) = overcast;
		data(0, 1) = rain;
		data(0, 2) = sunny;
		data(0, 3) = rain;
		data(0, 4) = sunny;

		data(1, 0) = hot;
		data(1, 1) = cool;
		data(1, 2) = mild;
		data(1, 3) = mild;
		data(1, 4) = hot;

		data(2, 0) = normal;
		data(2, 1) = high;
		data(2, 2) = high;
		data(2, 3) = normal;
		data(2, 4) = normal;

		data(3, 0) = strong;
		data(3, 1) = strong;
		data(3, 2) = weak;
		data(3, 3) = weak;
		data(3, 4) = strong;

		labels[0] = 1.0;
		labels[1] = 0.0;
		labels[2] = 0.0;
		labels[3] = 1.0;
		labels[3] = 1.0;
	}
}
