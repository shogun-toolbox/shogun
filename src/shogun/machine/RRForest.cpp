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

#include <shogun/machine/RRForest.h>
#include <shogun/multiclass/tree/RRCARTree.h>

using namespace shogun;

CRRForest::CRRForest()
: CBaggingMachine()
{
	init();
}

CRRForest::CRRForest(int32_t rand_numfeats, int32_t num_bags)
: CBaggingMachine()
{
	init();
	
	set_num_bags(num_bags);

	if (rand_numfeats>0)
		dynamic_cast<CRRCARTree*>(m_machine)->set_feature_subset_size(rand_numfeats);
}

CRRForest::CRRForest(CFeatures* features, CLabels* labels, int32_t num_bags, int32_t rand_numfeats)
: CBaggingMachine()
{
	init();

	SG_REF(features);
	m_features=features;
	set_labels(labels);

	set_num_bags(num_bags);

	if (rand_numfeats>0)
		dynamic_cast<CRRCARTree*>(m_machine)->set_feature_subset_size(rand_numfeats);
}

CRRForest::CRRForest(CFeatures* features, CLabels* labels, SGVector<float64_t> weights, int32_t num_bags, int32_t rand_numfeats)
: CBaggingMachine()
{
	init();

	SG_REF(features);
	m_features=features;
	set_labels(labels);
	m_weights=weights;

	set_num_bags(num_bags);

	if (rand_numfeats>0)
		dynamic_cast<CRRCARTree*>(m_machine)->set_feature_subset_size(rand_numfeats);
}

CRRForest::~CRRForest()
{
}

void CRRForest::set_machine(CMachine* machine)
{
	SG_ERROR("Machine is set as CRRCART and cannot be changed\n")
}

void CRRForest::set_weights(SGVector<float64_t> weights)
{
	m_weights=weights;
}

SGVector<float64_t> CRRForest::get_weights() const
{
	return m_weights;
}

EProblemType CRRForest::get_machine_problem_type() const
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RRCARTree\n")
	return dynamic_cast<CRRCARTree*>(m_machine)->get_machine_problem_type();
}

void CRRForest::set_machine_problem_type(EProblemType mode)
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RRCARTree\n")
	dynamic_cast<CRRCARTree*>(m_machine)->set_machine_problem_type(mode);
}

void CRRForest::set_num_random_features(int32_t rand_featsize)
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RRCARTree\n")
	REQUIRE(rand_featsize>0,"feature subset size should be greater than 0\n")

	dynamic_cast<CRRCARTree*>(m_machine)->set_feature_subset_size(rand_featsize);
}

int32_t CRRForest::get_num_random_features() const
{
	REQUIRE(m_machine,"m_machine is NULL. It is expected to be RRCARTree\n")
	return dynamic_cast<CRRCARTree*>(m_machine)->get_feature_subset_size();
}

void CRRForest::set_machine_parameters(CMachine* m, SGVector<index_t> idx)
{
	REQUIRE(m,"Machine supplied is NULL\n")
	REQUIRE(m_machine,"Reference Machine is NULL\n")

	CRRCARTree* tree=dynamic_cast<CRRCARTree*>(m);

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
	// equate the machine problem types - cloning does not do this
	tree->set_machine_problem_type(dynamic_cast<CRRCARTree*>(m_machine)->get_machine_problem_type());
}

bool CRRForest::train_machine(CFeatures* data)
{
	if (data)
	{
		SG_REF(data);
		SG_UNREF(m_features);
		m_features = data;
	}
	
	REQUIRE(m_features, "Training features not set!\n");
	
	return CBaggingMachine::train_machine();
}

void CRRForest::init()
{
	m_machine=new CRRCARTree();
	m_weights=SGVector<float64_t>();

	SG_ADD(&m_weights,"m_weights","weights",MS_NOT_AVAILABLE)
}

