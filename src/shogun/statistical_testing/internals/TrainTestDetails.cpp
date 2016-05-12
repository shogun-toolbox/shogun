/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/io/SGIO.h>
#include <shogun/statistical_testing/internals/TrainTestDetails.h>

using namespace shogun;
using namespace internal;

TrainTestDetails::TrainTestDetails() : m_total_num_samples(0), m_num_training_samples(0)
{
}

void TrainTestDetails::set_total_num_samples(index_t total_num_samples)
{
	m_total_num_samples=total_num_samples;
}

index_t TrainTestDetails::get_total_num_samples() const
{
	return m_total_num_samples;
}

void TrainTestDetails::set_num_training_samples(index_t num_training_samples)
{
	REQUIRE(m_total_num_samples>=num_training_samples,
			"Number of training samples cannot be greater than the total number of samples!\n");
	m_num_training_samples=num_training_samples;
}

index_t TrainTestDetails::get_num_training_samples() const
{
	return m_num_training_samples;
}

index_t TrainTestDetails::get_num_test_samples() const
{
	return m_total_num_samples-m_num_training_samples;
}
//
//bool TrainTestDetails::is_training_mode() const
//{
//}
//
//void TrainTestDetails::set_train_mode(bool train_mode)
//{
//}
//
//void TrainTestDetails::set_xvalidation_mode(bool xvalidation_mode)
//{
//}
