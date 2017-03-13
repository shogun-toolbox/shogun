/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
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
 * Written (W) 2016 Saurabh Mahindre
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/machine/RRForest.h>
#include <shogun/ensemble/MajorityVote.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <gtest/gtest.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/preprocessor/NormOne.h>

using namespace shogun;

#ifdef HAVE_LAPACK
void generate_multiclass_data(SGMatrix<float64_t>& feat, SGVector<float64_t>& lab,
	   	int32_t num, int32_t classes, int32_t feats)
{
	feat = CDataGenerator::generate_gaussians(num,classes,feats);
	for( int i = 0 ; i < classes ; ++i )
		for( int j = 0 ; j < num ; ++j )
			lab[i*num+j] = double(i);

}

TEST(RRForest, classify)
{
	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;
	CMath::init_random(1);

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);

	generate_multiclass_data(feat, lab, num, classes, feats);
	
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));

	//generate random subset for train and test data
	train.random(0, classes*num-1);
	test.random(0, classes*num-1);

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);
	CFeatures* features_test = (CFeatures*) features->clone();	
	CLabels* labels_test = (CLabels*) labels->clone();

	CRRForest* c=new CRRForest(features, labels, 100,2);
	CMajorityVote* mv = new CMajorityVote();
	c->set_combination_rule(mv);
	c->parallel->set_num_threads(1);	

	features->add_subset(train);
	labels->add_subset(train);	
	c->train(features);

	features_test->add_subset(test);
	labels_test->add_subset(test);
	CMulticlassLabels* output=CLabelsFactory::to_multiclass(c->apply(features_test));
	SG_REF(output);
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
	SG_UNREF(c);
	SG_UNREF(features_test);
	SG_UNREF(labels_test);
}
#endif

