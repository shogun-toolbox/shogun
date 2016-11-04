/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Saurabh Mahindre
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
 *
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <gtest/gtest.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;

void generate_data(SGMatrix<float64_t>& mat, SGVector<float64_t> &lab)
{
	int32_t num=lab.size();

	for (index_t i=0; i<num; i++)
	{
		mat(0,i)=i<num/2 ? 0+(CMath::randn_double()*4) : 100+(CMath::randn_double()*4)	;
		mat(1,i)=i;
	}

	for (index_t i=0; i<num; ++i)
		lab.vector[i]=i<num/2 ? 0 : 1;

}

TEST(CrossValidation_multithread, LibSVM_unlocked)
{
	int32_t num=100;
	SGMatrix<float64_t> mat(2, num);

	/* training labels +/- 1 for each cluster */
	SGVector<float64_t> lab(num);

	/*create simple linearly separable data*/
	generate_data(mat, lab);

	sg_rand->set_seed(1);

	for (index_t i=0; i<num/2; ++i)
		lab.vector[i]-=1;
	CBinaryLabels* labels=new CBinaryLabels(lab);

	CDenseFeatures<float64_t>* features=
			new CDenseFeatures<float64_t>(mat);
	SG_REF(features);	

	int32_t width=100;
	CGaussianKernel* kernel=new CGaussianKernel(width);
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=1;
	CLibSVM* svm=new CLibSVM(svm_C, kernel, labels);

	CContingencyTableEvaluation* eval_crit=
			new CContingencyTableEvaluation(ACCURACY);

	index_t n_folds=4;
	CStratifiedCrossValidationSplitting* splitting=
			new CStratifiedCrossValidationSplitting(labels, n_folds);

	CCrossValidation* cross=new CCrossValidation(svm, features, labels,
			splitting, eval_crit);

	cross->set_autolock(false);
	cross->set_num_runs(4);
	cross->parallel->set_num_threads(1);	

	CCrossValidationResult* result1=(CCrossValidationResult*)cross->evaluate();
	float64_t mean1 = result1->mean;
	
	cross->parallel->set_num_threads(3);

	CCrossValidationResult* result2=(CCrossValidationResult*)cross->evaluate();
	float64_t mean2 = result2->mean;

	EXPECT_EQ(mean1, mean2);

	/* clean up */
	SG_UNREF(result1);
	SG_UNREF(result2);
	SG_UNREF(cross);
	SG_UNREF(features);
}

TEST(CrossValidation_multithread, KNN)
{
	int32_t num=100;
	SGMatrix<float64_t> mat(2, num);

	SGVector<float64_t> lab(num);

	/*create simple linearly separable data*/
	generate_data(mat, lab);
	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CDenseFeatures<float64_t>* features=
			new CDenseFeatures<float64_t>(mat);
	SG_REF(features);	

	/* create knn */
	CEuclideanDistance* distance = new CEuclideanDistance(features, features);
	CKNN* knn=new CKNN (4, distance, labels);
	/* evaluation criterion */
	CMulticlassAccuracy* eval_crit = new CMulticlassAccuracy ();

	/* splitting strategy */
	index_t n_folds=4;
	CStratifiedCrossValidationSplitting* splitting=
			new CStratifiedCrossValidationSplitting(labels, n_folds);

	CCrossValidation* cross=new CCrossValidation(knn, features, labels,
			splitting, eval_crit);

	cross->set_autolock(false);
	cross->set_num_runs(4);
	cross->parallel->set_num_threads(1);	

	CCrossValidationResult* result1=(CCrossValidationResult*)cross->evaluate();
	float64_t mean1 = result1->mean;
	
	cross->parallel->set_num_threads(3);

	CCrossValidationResult* result2=(CCrossValidationResult*)cross->evaluate();
	float64_t mean2 = result2->mean;

	EXPECT_EQ(mean1, mean2);

	SG_UNREF(result1);
	SG_UNREF(result2);
	SG_UNREF(cross);
	SG_UNREF(features);
}
