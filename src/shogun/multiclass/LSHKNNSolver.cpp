/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Viktor Gal
 */

#include <shogun/multiclass/LSHKNNSolver.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/base/progress.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/Signal.h>

using namespace shogun;
using namespace Eigen;

#include <shogun/lib/external/falconn/lsh_nn_table.h>

CLSHKNNSolver::CLSHKNNSolver(const int32_t k, const float64_t q, const int32_t num_classes, const int32_t min_label, const SGVector<int32_t> train_labels, const int32_t lsh_l, const int32_t lsh_t):
CKNNSolver(k, q, num_classes, min_label, train_labels)
{
	init();

	m_lsh_l=lsh_l;
	m_lsh_t=lsh_t;
}

template<typename PointType, typename FeatureType>
PointType get_falconn_point(FeatureType* f, index_t i);

template<>
falconn::DenseVector<double> get_falconn_point(CDenseFeatures<float64_t>* f, index_t i)
{
	index_t len;
	bool free;
	float64_t* vec = f->get_feature_vector(i, len, free);
	return Map<VectorXd>(vec, len);
}

template<>
falconn::SparseVector<double> get_falconn_point(CSparseFeatures<float64_t>* f, index_t i)
{
	// FIXME: this basically copies the data :(
	auto fv = f->get_sparse_feature_vector(i);
	falconn::SparseVector<double> mapped(fv.num_feat_entries);
	for (index_t j = 0; j < fv.num_feat_entries; ++j)
		mapped[j] = std::make_pair(fv.features[j].feat_index, fv.features[j].entry);
	return mapped;
}

template<typename PointType, typename FeatureType>
CMulticlassLabels* CLSHKNNSolver::classify_objects(FeatureType* lhs, FeatureType* query_features, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const
{
	auto output = new CMulticlassLabels(num_lab);
	std::vector<PointType> feats(lhs->get_num_vectors());
	for(index_t i = 0; i < lhs->get_num_vectors(); ++i)
		feats[i] = get_falconn_point<PointType>(lhs, i);

	falconn::LSHConstructionParameters params
		= falconn::get_default_parameters<PointType>(lhs->get_num_vectors(),
                           lhs->get_num_features(),
                           falconn::DistanceFunction::EuclideanSquared,
                           true);
	SG_UNREF(lhs);
	if (m_lsh_l && m_lsh_t)
		params.l = m_lsh_l;

	auto lsh_table = falconn::construct_table<PointType>(feats, params);
	if (m_lsh_t)
		lsh_table->set_num_probes(m_lsh_t);

	SGMatrix<index_t> NN (m_k, query_features->get_num_vectors());
	for(index_t i = 0; i < query_features->get_num_vectors(); ++i)
	{
		auto indices = new std::vector<index_t> ();
		lsh_table->find_k_nearest_neighbors(get_falconn_point<PointType>(query_features, i), (int_fast64_t)m_k, indices);
		sg_memcpy(NN.get_column_vector(i), indices->data(), sizeof(index_t)*m_k);
		delete indices;
	}
	for (auto i: progress(range(num_lab)))
	{
		if(cancel_computation())break;
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

CMulticlassLabels* CLSHKNNSolver::classify_objects(CDistance* knn_distance, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<float64_t>& classes) const
{
	auto lhs = knn_distance->get_lhs();
	auto rhs = knn_distance->get_rhs();
	if ((lhs->get_feature_class() == C_DENSE) && (rhs->get_feature_class() == C_DENSE))
	{
		auto features = lhs->as<CDenseFeatures<float64_t>>();
		auto query_features = rhs->as<CDenseFeatures<float64_t>>();
		return classify_objects<falconn::DenseVector<double>>(features, query_features, num_lab, train_lab, classes);
	}
	else if ((lhs->get_feature_class() == C_SPARSE) && (rhs->get_feature_class() == C_SPARSE))
	{
		auto features = lhs->as<CSparseFeatures<float64_t>>();
		auto query_features = rhs->as<CSparseFeatures<float64_t>>();
		return classify_objects<falconn::SparseVector<double>>(features, query_features, num_lab, train_lab, classes);
	}
	else
	{
		SG_ERROR("Unsupported feature type!")
	}
}

SGVector<int32_t> CLSHKNNSolver::classify_objects_k(CDistance* d, const int32_t num_lab, SGVector<int32_t>& train_lab, SGVector<int32_t>& classes) const
{
	SG_NOTIMPLEMENTED
	return 0;
}
