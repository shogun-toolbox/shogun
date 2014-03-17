/***********************************************************************
 * Software License Agreement (BSD License)
 * Written (W) 2014 Dhruv Jawali (dhruv13.j@gmail.com)
 * Written (W) 2013 Fernando J. Iglesias García
 *   All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <gtest/gtest.h>

using namespace shogun;

void test_knn(CKNN::EKNNMode mode)
{
	int32_t dim = 2;
	int32_t num = 3;
	
	SGMatrix<float64_t> feat(dim, 4*num*num);
	int32_t x = 0, k = 3;
	for (int32_t i = -(2*num - 1); i < (2*num + 1); i+=2)
	{
		for (int32_t j = -(2*num-1); j < (2*num + 1); j+=2)
		{
			feat(0, x) = i;
			feat(1, x++) = j;
		}
	}
	CDenseFeatures<float64_t>* features = new CDenseFeatures<float64_t>(feat);
	
	SGVector<float64_t> lab(36);
	x = 0;
	for (int32_t i = -(2*num - 1); i < (2*num + 1); i+=2)
	{
		for (int32_t j = -(2*num-1); j < (2*num + 1); j+=2)
		{
			if ( i > 0 && j > 0 ) lab[x++] = 1;
			if ( i < 0 && j > 0 ) lab[x++] = 2;
			if ( i < 0 && j < 0 ) lab[x++] = 3;
			if ( i > 0 && j < 0 ) lab[x++] = 4;
		}
	}
	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	
	SGMatrix<float64_t> dat(2, 4);
	dat(0, 0) = 4.0;
	dat(1, 0) = 4.0;
	dat(0, 1) = -4.0;
	dat(1, 1) = 4.0;
	dat(0, 2) = -4.0;
	dat(1, 2) = -4.0;
	dat(0, 3) = 4.0;
	dat(1, 3) = -4.0;
	CDenseFeatures<float64_t>* data = new CDenseFeatures<float64_t>(dat);
	
	CKNN* knn = new CKNN(k, new CEuclideanDistance(features, features), labels, mode);
	knn->train();
	CMulticlassLabels* output = CLabelsFactory::to_multiclass( knn->apply(data) );
	
	for (int32_t i = 1; i <= 4; i++)
		EXPECT_EQ(i, output->get_label(i-1));
	
	SG_UNREF(knn)
	SG_UNREF(output)
}

TEST(KNN, BruteForce)
{
	test_knn(CKNN::BruteForce);
}

TEST(KNN, CoverTree)
{
	test_knn(CKNN::CoverTree);
}

#ifdef HAVE_NANOFLANN
TEST(KNN, KDTree)
{
	test_knn(CKNN::KDTree);
}
#endif
