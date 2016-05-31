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

#ifndef HYPOTHESIS_TEST_H_
#define HYPOTHESIS_TEST_H_

#include <memory>
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

class CFeatures;

namespace internal
{

class DataManager;

}

class CHypothesisTest : public CSGObject
{
public:
	explicit CHypothesisTest(index_t num_distributions);
	virtual ~CHypothesisTest();

	CHypothesisTest(const CHypothesisTest& other)=delete;
	CHypothesisTest& operator=(const CHypothesisTest& other)=delete;

	void set_train_test_mode(bool on);
	void set_train_test_ratio(float64_t ratio);

	virtual float64_t compute_p_value(float64_t statistic);
	virtual float64_t compute_threshold(float64_t alpha);
	virtual bool perform_test(float64_t alpha);

	virtual float64_t compute_statistic()=0;
	virtual SGVector<float64_t> sample_null()=0;

	virtual const char* get_name() const;
	virtual CSGObject* clone();
protected:
	internal::DataManager& get_data_mgr();
	const internal::DataManager& get_data_mgr() const;
private:
	struct Self;
	std::unique_ptr<Self> self;
};

}

#endif // HYPOTHESIS_TEST_H_
