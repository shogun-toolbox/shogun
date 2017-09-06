/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/multiclass/KDTreeKNNSolver.h>
#include <shogun/multiclass/tree/KDTree.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CKDTREEKNNSolver::CKDTREEKNNSolver(
    const index_t k, const float64_t q, const index_t num_classes,
    const index_t min_label, const SGVector<index_t> train_labels,
    const index_t leaf_size)
    : CKNNSolver(k, q, num_classes, min_label, train_labels)
{
	init();

	m_leaf_size=leaf_size;
}

CMulticlassLabels* CKDTREEKNNSolver::classify_objects(
    CDistance* knn_distance, const index_t num_lab,
    SGVector<index_t>& train_lab, SGVector<float64_t>& classes) const
{
	CMulticlassLabels* output=new CMulticlassLabels(num_lab);
	CFeatures* lhs = knn_distance->get_lhs();
	CKDTree* kd_tree = new CKDTree(m_leaf_size);
	kd_tree->build_tree(dynamic_cast<CDenseFeatures<float64_t>*>(lhs));
	SG_UNREF(lhs);

	CFeatures* query = knn_distance->get_rhs();
	kd_tree->query_knn(dynamic_cast<CDenseFeatures<float64_t>*>(query), m_k);
	SGMatrix<index_t> NN = kd_tree->get_knn_indices();
	for (index_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j = 0; j < m_k; j++)
			train_lab[j] = m_train_labels[ NN(j,i) ];

		//get the index of the 'nearest' class
		index_t out_idx = choose_class(classes.vector, train_lab.vector);
		//write the label of 'nearest' in the output
		output->set_label(i, out_idx + m_min_label);
	}
	SG_UNREF(query);
	SG_UNREF(kd_tree);
	return output;
}

SGVector<index_t> CKDTREEKNNSolver::classify_objects_k(
    CDistance* knn_distance, const index_t num_lab,
    SGVector<index_t>& train_lab, SGVector<index_t>& classes) const
{
	SGVector<index_t> output(m_k * num_lab);

	//allocation for distances to nearest neighbors
	SGVector<float64_t> dists(m_k);

	CFeatures* lhs = knn_distance->get_lhs();
	CKDTree* kd_tree = new CKDTree(m_leaf_size);
	kd_tree->build_tree(dynamic_cast<CDenseFeatures<float64_t>*>(lhs));
	SG_UNREF(lhs);

	CFeatures* data = knn_distance->get_rhs();
	kd_tree->query_knn(dynamic_cast<CDenseFeatures<float64_t>*>(data), m_k);
	SGMatrix<index_t> NN = kd_tree->get_knn_indices();
	for (index_t i = 0; i < num_lab && (!cancel_computation()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
		{
			train_lab[j] = m_train_labels[ NN(j,i) ];
			dists[j] = knn_distance->distance(NN(j,i), i);
		}
		CMath::qsort_index(dists.vector, train_lab.vector, m_k);

		choose_class_for_multiple_k(output.vector+i, classes.vector, train_lab.vector, num_lab);
	}

	SG_UNREF(data);
	SG_UNREF(kd_tree);
	
	return output;
}
