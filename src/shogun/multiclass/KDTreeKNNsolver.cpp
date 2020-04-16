/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/multiclass/KDTreeKNNSolver.h>
#include <shogun/multiclass/tree/KDTree.h>
#include <shogun/lib/Signal.h>

#include <iostream>

using namespace shogun;

KDTREEKNNSolver::KDTREEKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels,  const int32_t leaf_size):
KNNSolver(k, q, num_classes, min_label, train_labels)
{
	std::cout<<"entered KDTREEKNNSolver::KDTREEKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels,  const int32_t leaf_size)\n";
	init();

	m_leaf_size=leaf_size;
	std::cout<<"exiting KDTREEKNNSolver::KDTREEKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels,  const int32_t leaf_size)\n";
}

std::shared_ptr<MulticlassLabels> KDTREEKNNSolver::classify_objects(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes)
{
	std::cout<<"entered KDTREEKNNSolver::classify_objects(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes)\n";
	auto output=std::make_shared<MulticlassLabels>(num_lab);
	//auto lhs = knn_distance->get_lhs();
	//auto kd_tree = std::make_shared<KDTree>(m_leaf_size);
	//kd_tree->build_tree(lhs->as<DenseFeatures<float64_t>>());

	//auto query = knn_distance->get_rhs();
	//m_kd_tree->query_knn(query->as<DenseFeatures<float64_t>>(), m_k);
	//SGMatrix<index_t> NN = m_kd_tree->get_knn_indices();
	compute_nearest_neighbours(knn_distance);
	for (int32_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (int32_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ m_NN(j,i) ];

		//get the index of the 'nearest' class
		int32_t out_idx = choose_class(classes.vector, train_lab.vector);
		//write the label of 'nearest' in the output
		output->set_label(i, out_idx + m_min_label);
	}
	std::cout<<"exiting KDTREEKNNSolver::classify_objects(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes)\n";
	return output;
}

SGVector<int32_t> KDTREEKNNSolver::classify_objects_k(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes)
{
	std::cout<<"entered KDTREEKNNSolver::classify_objects_k(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes)\n";
	SGVector<int32_t> output(m_k*num_lab);

	//allocation for distances to nearest neighbors
	SGVector<float64_t> dists(m_k);

	//auto lhs = knn_distance->get_lhs();
	//auto kd_tree = std::make_shared<KDTree>(m_leaf_size);
	//kd_tree->build_tree(lhs->as<DenseFeatures<float64_t>>());

	//auto data = knn_distance->get_rhs();
	//m_kd_tree->query_knn(data->as<DenseFeatures<float64_t>>(), m_k);
	//SGMatrix<index_t> NN = m_kd_tree->get_knn_indices();
	compute_nearest_neighbours(knn_distance);
	for (index_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
		{
			train_lab[j] = m_train_labels[ m_NN(j,i) ];
			dists[j] = knn_distance->distance(m_NN(j,i), i);
		}
		Math::qsort_index(dists.vector, train_lab.vector, m_k);

		choose_class_for_multiple_k(output.vector+i, classes.vector, train_lab.vector, num_lab);
	}
	std::cout<<"exiting KDTREEKNNSolver::classify_objects_k(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes)\n";
	return output;
}

bool KDTREEKNNSolver::train_KNN(std::shared_ptr<Distance> knn_distance)
{
	std::cout<<"entered KDTREEKNNSolver::train_KNN(std::shared_ptr<Distance> knn_distance)\n";
	//m_knn_distance = knn_distance;
	m_kd_tree = std::make_shared<KDTree>(m_leaf_size);
	auto lhs = knn_distance->get_lhs();
	m_kd_tree->build_tree(lhs->as<DenseFeatures<float64_t>>());
	std::cout<<"exiting KDTREEKNNSolver::train_KNN(std::shared_ptr<Distance> knn_distance)\n";
	return true;
}

bool KDTREEKNNSolver::compute_nearest_neighbours(std::shared_ptr<Distance> knn_distance)
{
	std::cout<<"entered KDTREEKNNSolver::compute_nearest_neighbours()\n";
	auto query = knn_distance->get_rhs();
	m_kd_tree->query_knn(query->as<DenseFeatures<float64_t>>(), m_k);
	m_NN = m_kd_tree->get_knn_indices();
	std::cout<<"exiting KDTREEKNNSolver::compute_nearest_neighbours()\n";
	return true;
}
