/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 - 2017 Soumyajit De
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

#include <shogun/statistical_testing/OneDistributionTest.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/TestTypes.h>

using namespace shogun;
using namespace internal;

COneDistributionTest::COneDistributionTest() : CHypothesisTest(OneDistributionTest::num_feats)
{
}

COneDistributionTest::~COneDistributionTest()
{
}

void COneDistributionTest::set_samples(CFeatures* samples)
{
	auto& data_mgr=get_data_mgr();
	data_mgr.samples_at(0)=samples;
}

CFeatures* COneDistributionTest::get_samples() const
{
	const auto& data_mgr=get_data_mgr();
	return data_mgr.samples_at(0);
}

void COneDistributionTest::set_num_samples(index_t num_samples)
{
	auto& data_mgr=get_data_mgr();
	data_mgr.num_samples_at(0)=num_samples;
}

index_t COneDistributionTest::get_num_samples() const
{
	const auto& data_mgr=get_data_mgr();
	return data_mgr.num_samples_at(0);
}

const char* COneDistributionTest::get_name() const
{
	return "OneDistributionTest";
}
