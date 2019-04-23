/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/multiclass/KDTreeKNNSolver.h>
#include <shogun/multiclass/tree/KDTree.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

KDTREEKNNSolver::KDTREEKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels,  const int32_t leaf_size):
KNNSolver(k, q, num_classes, min_label, train_labels)
{
	init();

	m_leaf_size=leaf_size;
}

std::shared_ptr<MulticlassLabels> KDTREEKNNSolver::classify_objects(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const
{
	auto output=std::make_shared<MulticlassLabels>(num_lab);
	auto lhs = knn_distance->get_lhs();
	auto kd_tree = std::make_shared<KDTree>(m_leaf_size);
	kd_tree->build_tree(lhs->as<DenseFeatures<float64_t>>());

	auto query = knn_distance->get_rhs();
	kd_tree->query_knn(query->as<DenseFeatures<float64_t>>(), m_k);
	SGMatrix<index_t> NN = kd_tree->get_knn_indices();
	for (int32_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (int32_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ NN(j,i) ];

		//get the index of the 'nearest' class
		int32_t out_idx = choose_class(classes.vector, train_lab.vector);
		//write the label of 'nearest' in the output
		output->set_label(i, out_idx + m_min_label);
	}
	return output;
}

SGVector<int32_t> KDTREEKNNSolver::classify_objects_k(std::shared_ptr<Distance> knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const
{
	SGVector<int32_t> output(m_k*num_lab);

	//allocation for distances to nearest neighbors
	SGVector<float64_t> dists(m_k);

	auto lhs = knn_distance->get_lhs();
	auto kd_tree = std::make_shared<KDTree>(m_leaf_size);
	kd_tree->build_tree(lhs->as<DenseFeatures<float64_t>>());

	auto data = knn_distance->get_rhs();
	kd_tree->query_knn(data->as<DenseFeatures<float64_t>>(), m_k);
	SGMatrix<index_t> NN = kd_tree->get_knn_indices();
	for (index_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
		{
			train_lab[j] = m_train_labels[ NN(j,i) ];
			dists[j] = knn_distance->distance(NN(j,i), i);
		}
		Math::qsort_index(dists.vector, train_lab.vector, m_k);

		choose_class_for_multiple_k(output.vector+i, classes.vector, train_lab.vector, num_lab);
	}

	return output;
}
