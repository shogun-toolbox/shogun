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

#include <utility>

using namespace shogun;

RandomForest::RandomForest()
: BaggingMachine()
{
	init();
}

RandomForest::RandomForest(int32_t rand_numfeats, int32_t num_bags)
: BaggingMachine()
{
	init();


	set_num_bags(num_bags);

	if (rand_numfeats>0)
		m_machine->as<RandomCARTree>()->set_feature_subset_size(rand_numfeats);
}

RandomForest::RandomForest(std::shared_ptr<Features> features, std::shared_ptr<Labels> labels, int32_t num_bags, int32_t rand_numfeats)
: BaggingMachine()
{
	init();


	m_features=std::move(features);
	set_labels(std::move(labels));

	set_num_bags(num_bags);

	if (rand_numfeats>0)
		m_machine->as<RandomCARTree>()->set_feature_subset_size(rand_numfeats);
}

RandomForest::RandomForest(std::shared_ptr<Features> features, std::shared_ptr<Labels> labels, SGVector<float64_t> weights, int32_t num_bags, int32_t rand_numfeats)
: BaggingMachine()
{
	init();


	m_features=std::move(features);
	set_labels(std::move(labels));
	m_weights=weights;

	set_num_bags(num_bags);

	if (rand_numfeats>0)
		m_machine->as<RandomCARTree>()->set_feature_subset_size(rand_numfeats);
}

RandomForest::~RandomForest()
{
}

void RandomForest::set_machine(std::shared_ptr<Machine> machine)
{
	error("Machine is set as CRandomCART and cannot be changed");
}

void RandomForest::set_weights(SGVector<float64_t> weights)
{
	m_weights=weights;
}

SGVector<float64_t> RandomForest::get_weights() const
{
	return m_weights;
}

void RandomForest::set_feature_types(SGVector<bool> ft)
{
	require(m_machine,"m_machine is NULL. It is expected to be RandomCARTree");
	m_machine->as<RandomCARTree>()->set_feature_types(ft);
}

SGVector<bool> RandomForest::get_feature_types() const
{
	require(m_machine,"m_machine is NULL. It is expected to be RandomCARTree");
	return m_machine->as<RandomCARTree>()->get_feature_types();
}

EProblemType RandomForest::get_machine_problem_type() const
{
	require(m_machine,"m_machine is NULL. It is expected to be RandomCARTree");
	return m_machine->as<RandomCARTree>()->get_machine_problem_type();
}

void RandomForest::set_machine_problem_type(EProblemType mode)
{
	require(m_machine,"m_machine is NULL. It is expected to be RandomCARTree");
	m_machine->as<RandomCARTree>()->set_machine_problem_type(mode);
}

void RandomForest::set_num_random_features(int32_t rand_featsize)
{
	require(m_machine,"m_machine is NULL. It is expected to be RandomCARTree");
	require(rand_featsize>0,"feature subset size should be greater than 0");

	m_machine->as<RandomCARTree>()->set_feature_subset_size(rand_featsize);
}

int32_t RandomForest::get_num_random_features() const
{
	require(m_machine,"m_machine is NULL. It is expected to be RandomCARTree");
	return m_machine->as<RandomCARTree>()->get_feature_subset_size();
}

void RandomForest::set_machine_parameters(std::shared_ptr<Machine> m, SGVector<index_t> idx)
{
	require(m,"Machine supplied is NULL");
	require(m_machine,"Reference Machine is NULL");

	auto tree=m->as<RandomCARTree>();

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
	tree->set_sorted_features(m_sorted_transposed_feats, m_sorted_indices);
	// equate the machine problem types - cloning does not do this
	tree->set_machine_problem_type(m_machine->as<RandomCARTree>()->get_machine_problem_type());
}

bool RandomForest::train_machine(std::shared_ptr<Features> data)
{
	if (data)
	{


		m_features = data;
	}
	
	require(m_features, "Training features not set!");

	m_machine->as<RandomCARTree>()->pre_sort_features(m_features, m_sorted_transposed_feats, m_sorted_indices);

	return BaggingMachine::train_machine();
}

void RandomForest::init()
{
	m_machine=std::make_shared<RandomCARTree>();
	m_weights=SGVector<float64_t>();

	SG_ADD(&m_weights, kWeights, "weights");
}
