/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/base/progress.h>
#include <shogun/lib/Signal.h>
#include <shogun/multiclass/BruteKNNSolver.h>
#include <shogun/multiclass/KNNSolver.h>
#include <shogun/machine/DistanceMachine.h>

#include <iostream>
#include <algorithm>

using namespace shogun;

BruteKNNSolver::BruteKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels):
KNNSolver(k, q, num_classes, min_label, train_labels)
{
	std::cout<<"entered BruteKNNSolver::constructor\n";
	init();
}

std::shared_ptr<MulticlassLabels> BruteKNNSolver::classify_objects(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes)
{
	auto output=std::make_shared<MulticlassLabels>(num_lab);
	
	//get the k nearest neighbors of each example
	compute_nearest_neighbours(knn_distance);

	//from the indices to the nearest neighbors, compute the class labels
	for (auto i : SG_PROGRESS(range(num_lab)))
	{
		if (cancel_computation())
			break;
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ m_NN(j,i) ];

		//get the index of the 'nearest' class
		index_t out_idx = choose_class(classes.vector, train_lab.vector);
		//write the label of 'nearest' in the output
		output->set_label(i, out_idx + m_min_label);
	}

	return output;
}

SGVector<int32_t> BruteKNNSolver::classify_objects_k(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes)
{
	SGVector<int32_t> output(m_k*num_lab);

	//get the k nearest neighbors of each example
	compute_nearest_neighbours(knn_distance);

	for (index_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ m_NN(j,i) ];

		choose_class_for_multiple_k(output.vector+i, classes.vector, train_lab.vector, num_lab);
	}

	return output;
}

bool BruteKNNSolver::train_KNN(std::shared_ptr<Distance> knn_distance)
{
	//TODO complete this
	
	return true;
}

bool BruteKNNSolver::compute_nearest_neighbours(std::shared_ptr<Distance> knn_distance)
{
	std::cout<<"entered BruteKNNSolver::compute_nearest_neighbours()\n";
	
	//number of examples to which kNN is applied
	int32_t n = knn_distance->get_num_vec_rhs();

	std::cout << "n is " << n << '\n';
	std::cout << "k is " << m_k << '\n';

	//distances to train data
	SGVector<float64_t> dists(m_train_labels.vlen);
	//indices to train data
	SGVector<index_t> train_idxs(m_train_labels.vlen);

	for (int i = 0; i<n; i++)
	{
		//COMPUTATION_CONTROLLERS
		distances_lhs(dists,0,m_train_labels.vlen-1,i);

		train_idxs.range_fill(0);

		std::pair<float64_t, index_t> pairt[m_train_labels.vlen];
	
		std::cout<<"i is "<<i<<'\n';
		dists.display_vector("before sorting dists");
		train_idxs.display_vector("before sorting train_idxs");

		// Storing the respective array elements in pairs.
		for (int j = 0; j < m_train_labels.vlen; j++)
		{
			pairt[j].first = dists[j];
			pairt[j].second = train_idxs[j];
		}

		std::sort(pairt, pairt + m_train_labels.vlen);

		for (int j = 0; j < m_train_labels.vlen; j++)
		{
			dists[j] = pairt[j].first;
			train_idxs[j] = pairt[j].second;
		}

		SG_DEBUG("\nQuick sort query {}", i);
		SG_DEBUG("{}", train_idxs.to_string());

			
		dists.display_vector("after sorting dists");
		train_idxs.display_vector("after sorting train_idxs is");

		//only considering the first k elements
		SGVector<index_t> nearest_k_train_idxs(train_idxs.vector, m_k, false);

		nearest_k_train_idxs.display_vector("nearest_k_train_idxs");

		m_NN.set_column(i, nearest_k_train_idxs);
	}
	knn_distance->reset_precompute();
	std::cout<<"exiting BruteKNNSolver::compute_nearest_neighbours()\n";
	return true;
}
