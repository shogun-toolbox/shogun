/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/machine/RandomForest.h>
#include <shogun/multiclass/tree/RandomCARTree.h>

using namespace shogun;

CRandomForest::CRandomForest()
: CBaggingMachine()
{
	init();
}

CRandomForest::CRandomForest(int32_t rand_numfeats)
: CBaggingMachine()
{
	init();

	dynamic_cast<CRandomCARTree*>(m_machine)->set_feature_subset_size(rand_numfeats);
}

CRandomForest::CRandomForest(CFeatures* features, CLabels* labels, int32_t rand_numfeats)
: CBaggingMachine()
{
	init();

	SG_REF(features);
	m_features=features;
	set_labels(labels);
	dynamic_cast<CRandomCARTree*>(m_machine)->set_feature_subset_size(rand_numfeats);
}

CRandomForest::CRandomForest(CFeatures* features, CLabels* labels, SGVector<float64_t> weights, int32_t rand_numfeats)
: CBaggingMachine()
{
	init();

	SG_REF(features);
	m_features=features;
	set_labels(labels);
	dynamic_cast<CRandomCARTree*>(m_machine)->set_weights(weights);
	dynamic_cast<CRandomCARTree*>(m_machine)->set_feature_subset_size(rand_numfeats);
}

CRandomForest::~CRandomForest()
{
}

void CRandomForest::set_bag_size(int32_t bag_size)
{
	SG_ERROR("Bag Size is set to be equal to number of training vectors and cannot be changed\n")
}

int32_t CRandomForest::get_bag_size() const
{
	REQUIRE(m_features,"Training features not set yet. Bag size cannot be determined\n")
	return (CDenseFeatures<float64_t>::obtain_from_generic(m_features))->get_num_vectors();
}

void CRandomForest::set_machine(CMachine* machine)
{
	SG_ERROR("Machine is set as CRandomCART and cannot be changed\n")
}

void CRandomForest::set_weights(SGVector<float64_t> weights)
{
	m_weights=weights;
}

SGVector<float64_t> CRandomForest::get_weights() const
{
	return m_weights;
}

void CRandomForest::set_feature_types(SGVector<bool> ft)
{
	dynamic_cast<CRandomCARTree*>(m_machine)->set_feature_types(ft);
}

SGVector<bool> CRandomForest::get_feature_types() const
{
	return dynamic_cast<CRandomCARTree*>(m_machine)->get_feature_types();
}

EProblemType CRandomForest::get_machine_problem_type() const
{
	return dynamic_cast<CRandomCARTree*>(m_machine)->get_machine_problem_type();
}

void CRandomForest::set_machine_problem_type(EProblemType mode)
{
	dynamic_cast<CRandomCARTree*>(m_machine)->set_machine_problem_type(mode);
}

void CRandomForest::set_num_random_features(int32_t rand_featsize)
{
	dynamic_cast<CRandomCARTree*>(m_machine)->set_feature_subset_size(rand_featsize);
}

int32_t CRandomForest::get_num_random_features() const
{
	return dynamic_cast<CRandomCARTree*>(m_machine)->get_feature_subset_size();
} 

void CRandomForest::set_machine_parameters(CMachine* m, SGVector<index_t> idx)
{
	REQUIRE(m,"Machine supplied is NULL\n")
	CRandomCARTree* tree=dynamic_cast<CRandomCARTree*>(m);

	SGVector<float64_t> weights(idx.vlen);

	if (m_weights.vlen==0)
	{
		weights.fill_vector(weights.vector,weights.vlen,1.0);
	}
	else
	{
		for (int32_t i=0;i<idx.vlen;i++)
			weights[i]=m_weights[idx[i]];
	}

	tree->set_weights(weights);
}

void CRandomForest::init()
{
	m_machine=new CRandomCARTree();
	SG_REF(m_machine);
}
