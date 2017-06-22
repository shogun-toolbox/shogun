/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: 2016 MikeLing, Viktor Gal, Sergey Lisitsyn, Heiko Strathmann
 */

#include <gtest/gtest.h>
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>

#include "environments/LinearTestEnvironment.h"

using namespace shogun;
#ifdef USE_SVMLIGHT
TEST(SVMLight, train)
{
	auto C = 1.0;
	auto epsilon = 0.001;
	std::shared_ptr<GaussianCheckerboard> mockData =
	    LinearTestEnvironment::instance().getBinaryLabelData();

	CDenseFeatures<float64_t>* train_feats = mockData->get_features_train();
	CDenseFeatures<float64_t>* test_feats = mockData->get_features_test();

	CBinaryLabels* ground_truth = (CBinaryLabels*)mockData->get_labels_train();

	CGaussianKernel* gauss_kernel =
	    new CGaussianKernel(train_feats, train_feats, 15);
	CSVMLight* svml = new CSVMLight(C, gauss_kernel, ground_truth);

	svml->set_epsilon(epsilon);
	svml->train();

	CLabels* pred = svml->apply(test_feats);

	CAccuracyMeasure evaluate = CAccuracyMeasure();
	evaluate.evaluate(pred, mockData->get_labels_test());
	EXPECT_GT(evaluate.get_accuracy(), 0.99);

	SG_UNREF(svml);
	SG_UNREF(pred);
}
#endif // USE_SVMLIGHT
