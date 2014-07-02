/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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

#include <shogun/lib/SGMatrix.h>
#include <shogun/labels/Labels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/statistics/IndependenceTest.h>
#include <shogun/preprocessor/DependenceMaximization.h>

using namespace shogun;

CDependenceMaximization::CDependenceMaximization()
	: CFeatureSelection<float64_t>()
{
	init();
}

void CDependenceMaximization::init()
{
	SG_ADD((CSGObject**)&m_estimator, "estimator",
			"the estimator for computing measures", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_labels_feats, "labels_feats",
			"the features based on labels", MS_NOT_AVAILABLE);

	m_estimator=NULL;
	m_labels_feats=NULL;
}

CDependenceMaximization::~CDependenceMaximization()
{
	SG_UNREF(m_estimator);
	SG_UNREF(m_labels_feats);
}

bool CDependenceMaximization::init(CFeatures* features)
{
	REQUIRE(features, "Features are not initialized!\n");
	REQUIRE(features->get_feature_class()==C_DENSE ||
			features->get_feature_class()==C_SPARSE,
			"Only allowed for dense/sparse features! Provided an instance of "
			"%s which is of class %d!\n",
			features->get_name(), features->get_feature_class());
	REQUIRE(features->get_feature_type()==F_DREAL, "Only allowed for "
			"features of double type! Provided %d!\n",
			features->get_feature_type());

	return true;
}

CFeatures* CDependenceMaximization::create_transformed_copy(CFeatures* features,
		index_t idx)
{
	SG_DEBUG("Entering!\n");

	// remove the dimension specified by the index, i.e. get X\X_i
	// NULL check is handled in CFeatureSelection::get_num_features call
	index_t num_features=get_num_features(features);
	REQUIRE(num_features>idx, "Specified dimension to remove (%d) is greater "
			"than the total number of current features (%d)!\n",
			idx, num_features);

	SGVector<index_t> dims(num_features-1);
	index_t n_dims=0;
	for (index_t i=0; i<num_features; ++i)
	{
		if (i!=idx)
			dims[n_dims++]=i;
	}

	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		dims.display_vector("dims");

	// the following already does a SG_REF on the newly created feature
	SG_DEBUG("Leaving!\n");
	return features->copy_dimension_subset(dims);
}

float64_t CDependenceMaximization::compute_measures(CFeatures* features,
		index_t idx)
{
	SG_DEBUG("Entering!\n");

	// remove the dimension (feat) specified by the index idx
	CFeatures* reduced_feats=create_transformed_copy(features, idx);
	ASSERT(reduced_feats);

	// perform an independence test for X\X_i ~ p and Y ~ q with
	// H_0: P(X\X_i, Y) = P(X\X_i) * P(Y)
	// the test statistic can then be used as a measure of dependence
	// See CIndependenceTest class documentation for details
	m_estimator->set_p(reduced_feats);
	float64_t statistic=m_estimator->compute_statistic();

	SG_DEBUG("statistic = %f!\n", statistic);

	SG_UNREF(reduced_feats);

	SG_DEBUG("Leaving!\n");
	return statistic;
}

CFeatures* CDependenceMaximization::remove_feats(CFeatures* features,
		SGVector<index_t> argsorted)
{
	SG_DEBUG("Entering!\n");

	REQUIRE(m_num_remove>0, "Number or percentage of features to be removed is "
			"not set! Please use set_num_remove() to set this!\n");
	REQUIRE(m_policy==N_SMALLEST || m_policy==PERCENTILE_SMALLEST,
			"Only N_SMALLEST and PERCENTILE_SMALLEST removal policy can work "
			"with %s!\n", get_name());
	REQUIRE(features, "Features is not intialized!\n");
	REQUIRE(argsorted.vector, "The argsorted vector is not initialized!\n");
	REQUIRE(get_num_features(features)==argsorted.vlen,
			"argsorted vector should be equal to the number of features (%d)! "
			"But it was %d!\n", argsorted.vlen);

	// compute a threshold to remove for both the policies
	index_t threshold=m_num_remove;
	if (m_policy==PERCENTILE_SMALLEST)
		threshold*=argsorted.vlen*0.01;

	// make sure that the threshold is valid given the current number of feats
	REQUIRE(threshold<argsorted.vlen, "The threshold of removal is too high "
			"(asked to remove %d features out of %d)! Please use a smaller "
			"number for removal using set_num_remove() call",
			threshold, argsorted.vlen);

	// remove the lowest threshold rank holders by storing indices
	SGVector<index_t> inds(argsorted.vlen-threshold);
	memcpy(inds.vector, argsorted.vector+threshold, sizeof(index_t)*inds.vlen);

	// sorting the indices to get the original order
	inds.qsort();
	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		inds.display_vector("selected feats");

	// copy rest of the features and SG_UNREF the original feat obj
	CFeatures* reduced_feats=features->copy_dimension_subset(inds);
	SG_UNREF(features);

	SG_DEBUG("Leaving!\n");
	return reduced_feats;
}

void CDependenceMaximization::set_policy(EFeatureRemovalPolicy policy)
{
	REQUIRE(policy==N_SMALLEST || policy==PERCENTILE_SMALLEST,
			"Only N_SMALLEST and PERCENTILE_SMALLEST removal policy can work "
			"with %s!\n", get_name());
	m_policy=policy;
}

void CDependenceMaximization::set_labels(CLabels* labels)
{
	// NULL check is handled in base class CFeatureSelection
	CFeatureSelection::set_labels(labels);

	// convert the CLabels object to CDenseFeatures
	SG_UNREF(m_labels_feats);

	SGMatrix<float64_t> labels_matrix(1, m_labels->get_num_labels());
	for (index_t i=0; i<labels_matrix.num_cols; ++i)
		labels_matrix.matrix[i]=m_labels->get_value(i);

	m_labels_feats=new CDenseFeatures<float64_t>(labels_matrix);
	SG_REF(m_labels_feats);

	// we need to set this to the estimator which is set internally
	ASSERT(m_estimator);
	m_estimator->set_q(m_labels_feats);
}
