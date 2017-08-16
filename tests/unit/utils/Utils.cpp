/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
*/
#include "Utils.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <shogun/io/SGIO.h>
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
