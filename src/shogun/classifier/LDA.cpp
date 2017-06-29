/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2014 Abhijeet Kislay
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#include <shogun/lib/config.h>

#include <shogun/classifier/LDA.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/common.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/FisherLDA.h>
#include <vector>

using namespace Eigen;
using namespace shogun;

CLDA::CLDA(float64_t gamma, ELDAMethod method)
	:CLinearMachine()
{
	init();
	m_method=method;
	m_gamma=gamma;
}

CLDA::CLDA(
    float64_t gamma, CDenseFeatures<float64_t>* traindat, CLabels* trainlab,
    ELDAMethod method)
    : CLinearMachine(), m_gamma(gamma)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
	m_method=method;
	m_gamma=gamma;
}

void CLDA::init()
{
	m_method=AUTO_LDA;
	m_gamma=0;
	SG_ADD(
	    (machine_int_t*)&m_method, "m_method",
	    "Method used for LDA calculation", MS_NOT_AVAILABLE);
	SG_ADD(
	    (machine_int_t*)&m_gamma, "m_gamma", "Regularization parameter",
	    MS_NOT_AVAILABLE);
}

CLDA::~CLDA()
{
}

bool CLDA::train_machine(CFeatures *data)
{
	REQUIRE(m_labels, "Labels for the given features are not specified!\n")
	REQUIRE(
	    m_labels->get_label_type() == LT_BINARY,
	    "The labels should of type"
	    " CBinaryLabels! you provided %s \n",
	    m_labels->get_name())

	if(data)
	{
		if(!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}
	else
	{
		data = get_features();
		REQUIRE(data, "Features have not been provided.\n")
	}

	SGVector<int32_t>train_labels=((CBinaryLabels *)m_labels)->get_int_labels();
	REQUIRE(train_labels.vector,"Provided Labels are empty!\n")

	REQUIRE(data->get_num_vectors() == train_labels.vlen,"Number of training examples(%d) should be "
		"equal to number of labels (%d)!\n", data->get_num_vectors(), train_labels.vlen);

	if(data->get_feature_type() == F_SHORTREAL)
		return CLDA::train_machine_templated<float32_t>(train_labels, data);
	else if(data->get_feature_type() == F_DREAL)
		return CLDA::train_machine_templated<float64_t>(train_labels, data);
	else if(data->get_feature_type() == F_LONGREAL)
		return CLDA::train_machine_templated<floatmax_t>(train_labels, data);

	return false;
}

template <typename ST>
bool CLDA::train_machine_templated(SGVector<int32_t> train_labels, CFeatures *data)
{
	SGMatrix<ST> feature_matrix =
	    ((CDenseFeatures<ST>*)data)->get_feature_matrix();
	index_t num_feat = feature_matrix.num_rows;
	index_t num_vec = feature_matrix.num_cols;

	bool lda_more_efficient = (m_method == AUTO_LDA && num_vec <= num_feat);

	if (m_method == SVD_LDA || lda_more_efficient)
		return solver_svd(train_labels, data);
	else
		return solver_classic<ST>(train_labels, data);
}

bool CLDA::solver_svd(SGVector<int32_t> train_labels, CFeatures* data)
{
	std::unique_ptr<CFisherLDA> lda(new CFisherLDA(CANVAR_FLDA));
	std::unique_ptr<CMulticlassLabels> multiclass_labels(
	    new CMulticlassLabels(m_labels->get_num_labels()));

	for (index_t i = 0; i < m_labels->get_num_labels(); ++i)
		multiclass_labels->set_int_label(
		    i, (((CBinaryLabels*)m_labels)->get_int_label(i) == 1 ? 1 : 0));

	// keep just the first dimension to do binary classification
	lda->fit(data, multiclass_labels.get(), 1);
	auto m = lda->get_transformation_matrix();

	SGVector<float64_t> w(m);
	set_w(w);
	set_bias(-linalg::dot(w, lda->get_mean()));

	return true;
}

template <typename ST>
bool CLDA::solver_classic(SGVector<int32_t> train_labels, CFeatures* data)
{
	SGMatrix<ST> feature_matrix =
	    ((CDenseFeatures<ST>*)data)->get_feature_matrix();
	index_t num_feat = feature_matrix.num_rows;
	index_t num_vec = feature_matrix.num_cols;

	std::vector<index_t> idx_neg;
	std::vector<index_t> idx_pos;

	for (index_t i = 0; i < train_labels.vlen; i++)
	{
		if (train_labels.vector[i] == -1)
			idx_neg.push_back(i);
		else if (train_labels.vector[i] == +1)
			idx_pos.push_back(i);
	}

	SGMatrix<ST> matrix(feature_matrix);
	SGVector<ST> mean_neg(num_feat);
	SGVector<ST> mean_pos(num_feat);

	linalg::zero(mean_neg);
	linalg::zero(mean_pos);

	// mean neg
	for (auto i : idx_neg)
		linalg::add_col_vec(matrix, i, mean_neg, mean_neg);
	linalg::scale(mean_neg, mean_neg, 1 / (ST)idx_neg.size());

	// get m(-ve) - mean(-ve)
	for (auto i : idx_neg)
		linalg::add_col_vec(matrix, i, mean_neg, matrix, (ST)1, (ST)-1);

	// mean pos
	for (auto i : idx_pos)
		linalg::add_col_vec(matrix, i, mean_pos, mean_pos);
	linalg::scale(mean_pos, mean_pos, 1 / (ST)idx_pos.size());

	// get m(+ve) - mean(+ve)
	for (auto i : idx_pos)
		linalg::add_col_vec(matrix, i, mean_pos, matrix, (ST)1, (ST)-1);

	// covariance matrix.
	auto cov_mat = linalg::matrix_prod(matrix, matrix, false, true);
	SGMatrix<ST> scatter_matrix(num_feat, num_feat);
	linalg::scale(cov_mat, scatter_matrix, 1 / (ST)(num_vec - 1));

	ST trace = linalg::trace(scatter_matrix);
	SGMatrix<ST> id(num_feat, num_feat);
	linalg::identity(id);
	linalg::add(
	    scatter_matrix, id, scatter_matrix, (ST)(1.0 - m_gamma),
	    trace * ((ST)m_gamma) / num_feat);

	// the usual way
	// we need to find a Basic Linear Solution of A.x=b for 'x'.
	// Instead of crudely Inverting A, we go for solve() using Decompositions.
	// where:
	// MatrixXd A=scatter;
	// VectorXd b=mean_pos-mean_neg;
	// VectorXd x=w;
	auto decomposition = linalg::cholesky_factor(scatter_matrix);
	SGVector<ST> w_st = linalg::cholesky_solver(
	    decomposition, linalg::add(mean_pos, mean_neg, (ST)1, (ST)-1));

	// get the weights w_neg(for -ve class) and w_pos(for +ve class)
	auto w_neg = linalg::cholesky_solver(decomposition, mean_neg);
	auto w_pos = linalg::cholesky_solver(decomposition, mean_pos);

	SGVector<float64_t> w(num_feat);
	// copy w_st into w
	for (index_t i = 0; i < w.size(); ++i)
		w[i] = (float64_t)w_st[i];
	set_w(w);

	// get the bias.
	set_bias(
	    (float64_t)(
	        0.5 *
	        (linalg::dot(w_neg, mean_neg) - linalg::dot(w_pos, mean_pos))));

	return true;
}
