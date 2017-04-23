/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#include <shogun/multiclass/LSHKNNSolver.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/Signal.h>

using namespace shogun;
using namespace Eigen;

#ifdef HAVE_CXX11
#include <shogun/lib/external/falconn/lsh_nn_table.h>

CLSHKNNSolver::CLSHKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels, const int32_t lsh_l, const int32_t lsh_t):
CKNNSolver(k, q, num_classes, min_label, train_labels)
{
	init();

	m_lsh_l=lsh_l; 
	m_lsh_t=lsh_t;
}

CMulticlassLabels* CLSHKNNSolver::classify_objects(CDistance* knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const
{
	CMulticlassLabels* output=new CMulticlassLabels(num_lab);
	CDenseFeatures<float64_t>* features = dynamic_cast<CDenseFeatures<float64_t>*>(knn_distance->get_lhs());
	std::vector<falconn::DenseVector<double>> feats;
	for(int32_t i=0; i < features->get_num_vectors(); i++)
	{
		int32_t len;
		bool free;
		float64_t* vec = features->get_feature_vector(i, len, free);
		falconn::DenseVector<double> temp = Map<VectorXd> (vec, len);
		feats.push_back(temp);
	}

	falconn::LSHConstructionParameters params 
		= falconn::get_default_parameters<falconn::DenseVector<double>>(features->get_num_vectors(),
                           features->get_num_features(),
                           falconn::DistanceFunction::EuclideanSquared,
                           true);
	SG_UNREF(features);
	if (m_lsh_l && m_lsh_t)
		params.l = m_lsh_l;

	auto lsh_table = falconn::construct_table<falconn::DenseVector<double>>(feats, params);
	if (m_lsh_t)
		lsh_table->set_num_probes(m_lsh_t);

	CDenseFeatures<float64_t>* query_features = dynamic_cast<CDenseFeatures<float64_t>*>(knn_distance->get_rhs());
	std::vector<falconn::DenseVector<double>> query_feats;

	SGMatrix<index_t> NN (m_k, query_features->get_num_vectors());
	for(index_t i=0; i < query_features->get_num_vectors(); i++)
	{
		int32_t len;
		bool free;
		float64_t* vec = query_features->get_feature_vector(i, len, free);
		falconn::DenseVector<double> temp = Map<VectorXd> (vec, len);
		auto indices = new std::vector<int32_t> ();
		lsh_table->find_k_nearest_neighbors(temp, (int_fast64_t)m_k, indices);
		sg_memcpy(NN.get_column_vector(i), indices->data(), sizeof(int32_t)*m_k);
		delete indices;
	}
		
	for (index_t i=0; i<num_lab && (!CSignal::cancel_computations()); i++)
	{
		//write the labels of the k nearest neighbors from theirs indices
		for (index_t j=0; j<m_k; j++)
			train_lab[j] = m_train_labels[ NN(j,i) ];

		//get the index of the 'nearest' class
		index_t out_idx = choose_class(classes.vector, train_lab.vector);
		//write the label of 'nearest' in the output
		output->set_label(i, out_idx + m_min_label);
	}
	SG_UNREF(query_features);

	return output;
}

SGVector<int32_t> CLSHKNNSolver::classify_objects_k(CDistance* d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const
{
	SG_NOTIMPLEMENTED
	return 0;
}
#endif
