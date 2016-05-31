/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#include <algorithm>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/statistical_testing/HypothesisTest.h>
#include <shogun/statistical_testing/internals/DataManager.h>

using namespace shogun;
using namespace internal;

struct CHypothesisTest::Self
{
	explicit Self(index_t num_distributions);
	DataManager data_mgr;
};

CHypothesisTest::Self::Self(index_t num_distributions) : data_mgr(num_distributions)
{
}

CHypothesisTest::CHypothesisTest(index_t num_distributions) : CSGObject()
{
	self=std::unique_ptr<Self>(new CHypothesisTest::Self(num_distributions));
}

CHypothesisTest::~CHypothesisTest()
{
}

void CHypothesisTest::set_train_test_mode(bool on)
{
	self->data_mgr.set_train_test_mode(on);
}

void CHypothesisTest::set_train_test_ratio(float64_t ratio)
{
	self->data_mgr.train_test_ratio(ratio);
}

float64_t CHypothesisTest::compute_p_value(float64_t statistic)
{
	SGVector<float64_t> values=sample_null();
	std::sort(values.vector, values.vector + values.vlen);
	float64_t i=values.find_position_to_insert(statistic);
	return 1.0-i/values.vlen;
}

float64_t CHypothesisTest::compute_threshold(float64_t alpha)
{
	SGVector<float64_t> values=sample_null();
	std::sort(values.vector, values.vector + values.vlen);
	return values[index_t(CMath::floor(values.vlen*(1-alpha)))];
}

bool CHypothesisTest::perform_test(float64_t alpha)
{
	auto statistic=compute_statistic();
	auto p_value=compute_p_value(statistic);
	return p_value<alpha;
}

const char* CHypothesisTest::get_name() const
{
	return "HypothesisTest";
}

CSGObject* CHypothesisTest::clone()
{
	SG_ERROR("Cloning CHypothesisTest instances is not supported!\n");
	return nullptr;
}

DataManager& CHypothesisTest::get_data_mgr()
{
	return self->data_mgr;
}

const DataManager& CHypothesisTest::get_data_mgr() const
{
	return self->data_mgr;
}
