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
#include <shogun/statistical_testing/IndependenceTest.h>
#include <shogun/preprocessor/DependenceMaximization.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

DependenceMaximization::DependenceMaximization()
	: FeatureSelection<float64_t>()
{
	init();
}

void DependenceMaximization::init()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_estimator, "estimator",
			"the estimator for computing measures");
	SG_ADD((std::shared_ptr<SGObject>*)&m_labels_feats, "labels_feats",
			"the features based on labels");

	m_estimator=NULL;
	m_labels_feats=NULL;
}

DependenceMaximization::~DependenceMaximization()
{


}

std::shared_ptr<Features> DependenceMaximization::create_transformed_copy(std::shared_ptr<Features> features,
		index_t idx)
{
	SG_TRACE("Entering!");

	// remove the dimension specified by the index, i.e. get X\X_i
	// NULL check is handled in FeatureSelection::get_num_features call
	index_t num_features=get_num_features(features);
	require(num_features>idx, "Specified dimension to remove ({}) is greater "
			"than the total number of current features ({})!",
			idx, num_features);

	SGVector<index_t> dims(num_features-1);
	index_t n_dims=0;
	for (index_t i=0; i<num_features; ++i)
	{
		if (i!=idx)
			dims[n_dims++]=i;
	}

	if (env()->io()->get_loglevel()<=io::MSG_DEBUG)
		dims.display_vector("dims");

	// the following already does a SG_REF on the newly created feature
	SG_TRACE("Leaving!");
	return features->copy_dimension_subset(dims);
}

float64_t DependenceMaximization::compute_measures(std::shared_ptr<Features> features,
		index_t idx)
{
	SG_TRACE("Entering!");

	// remove the dimension (feat) specified by the index idx
	auto reduced_feats=create_transformed_copy(features, idx);
	ASSERT(reduced_feats);

	// perform an independence test for X\X_i ~ p and Y ~ q with
	// H_0: P(X\X_i, Y) = P(X\X_i) * P(Y)
	// the test statistic can then be used as a measure of dependence
	// See IndependenceTest class documentation for details
	m_estimator->set_p(reduced_feats);
	float64_t statistic=m_estimator->compute_statistic();

	SG_DEBUG("statistic = {}!", statistic);



	SG_TRACE("Leaving!");
	return statistic;
}

std::shared_ptr<Features> DependenceMaximization::remove_feats(std::shared_ptr<Features> features,
		SGVector<index_t> argsorted)
{
	SG_TRACE("Entering!");

	require(m_num_remove>0, "Number or percentage of features to be removed is "
			"not set! Please use set_num_remove() to set this!");
	require(m_policy==N_LARGEST || m_policy==PERCENTILE_LARGEST,
			"Only N_LARGEST and PERCENTILE_LARGEST removal policy can work "
			"with {}!", get_name());
	require(features, "Features is not intialized!");
	require(argsorted.vector, "The argsorted vector is not initialized!");
	require(get_num_features(features)==argsorted.vlen,
			"argsorted vector should be equal to the number of features ({})! "
			"But it was {}!", argsorted.vlen);

	// compute a threshold to remove for both the policies
	index_t threshold=m_num_remove;
	if (m_policy==PERCENTILE_LARGEST)
		threshold*=argsorted.vlen*0.01;

	// make sure that the threshold is valid given the current number of feats
	require(threshold<argsorted.vlen, "The threshold of removal is too high "
			"(asked to remove {} features out of {})! Please use a smaller "
			"number for removal using set_num_remove() call",
			threshold, argsorted.vlen);

	// remove the highest rank holders by storing indices
	SGVector<index_t> inds(argsorted.vlen-threshold);
	sg_memcpy(inds.vector, argsorted.vector, sizeof(index_t)*inds.vlen);

	// sorting the indices to get the original order
	Math::qsort(inds);
	if (env()->io()->get_loglevel()<=io::MSG_DEBUG)
		inds.display_vector("selected feats");

	// copy rest of the features and SG_UNREF the original feat obj
	auto reduced_feats=features->copy_dimension_subset(inds);

	// add the selected features to the subset
	ASSERT(m_subset)
	m_subset->add_subset(inds);



	SG_TRACE("Leaving!");
	return reduced_feats;
}

void DependenceMaximization::set_policy(EFeatureRemovalPolicy policy)
{
	require(policy==N_LARGEST || policy==PERCENTILE_LARGEST,
			"Only N_LARGEST and PERCENTILE_LARGEST removal policy can work "
			"with {}!", get_name());
	m_policy=policy;
}

void DependenceMaximization::set_labels(std::shared_ptr<Labels> labels)
{
	// NULL check is handled in base class FeatureSelection
	FeatureSelection<float64_t>::set_labels(labels);

	// convert the Labels object to DenseFeatures


	SGMatrix<float64_t> labels_matrix(1, m_labels->get_num_labels());
	for (index_t i=0; i<labels_matrix.num_cols; ++i)
		labels_matrix.matrix[i]=m_labels->get_value(i);

	m_labels_feats=std::make_shared<DenseFeatures<float64_t>>(labels_matrix);


	// we need to set this to the estimator which is set internally
	ASSERT(m_estimator);
	m_estimator->set_q(m_labels_feats);
}
