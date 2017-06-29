/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Abhijeet Kislay
 */
#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/preprocessor/FisherLDA.h>

using namespace std;
using namespace Eigen;
using namespace shogun;

CFisherLDA::CFisherLDA(EFLDAMethod method, float64_t thresh, float64_t gamma)
    : CDimensionReductionPreprocessor()
{
	initialize_parameters();
	m_method=method;
	m_threshold=thresh;
	m_gamma = gamma;
}

void CFisherLDA::initialize_parameters()
{
	m_method=AUTO_FLDA;
	m_threshold=0.01;
	m_num_dim=0;
	m_gamma = 0;
	SG_ADD(
	    &m_method, "FLDA_method", "method for performing FLDA",
	    MS_NOT_AVAILABLE);
	SG_ADD(
	    &m_num_dim, "final_dimensions", "dimensions to be retained",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_gamma, "m_gamma", "Regularization parameter", MS_NOT_AVAILABLE);
	SG_ADD(
	    &m_transformation_matrix, "transformation_matrix",
	    "Transformation"
	    " matrix (Eigenvectors of covariance matrix).",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.", MS_NOT_AVAILABLE);
	SG_ADD(
	    &m_eigenvalues_vector, "eigenvalues_vector", "Vector with Eigenvalues.",
	    MS_NOT_AVAILABLE);
}

CFisherLDA::~CFisherLDA()
{
}

bool CFisherLDA::fit(CFeatures *features, CLabels *labels, int32_t num_dimensions)
{
	REQUIRE(features, "Features are not provided!\n")

	REQUIRE(features->get_feature_class()==C_DENSE,
			"LDA only works with dense features. you provided %s\n",
			features->get_name());

	REQUIRE(features->get_feature_type()==F_DREAL,
			"LDA only works with real features.\n");

	REQUIRE(labels, "Labels for the given features are not specified!\n")

	REQUIRE(
	    labels->get_label_type() == LT_MULTICLASS,
	    "The labels should be of "
	    "the type MulticlassLabels! you provided %s\n",
	    labels->get_name());

	SGMatrix<float64_t> feature_matrix =
	    ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	SGVector<float64_t> labels_vector=((CMulticlassLabels*)labels)->get_labels();

	index_t num_vectors = feature_matrix.num_cols;
	index_t num_features = feature_matrix.num_rows;

	REQUIRE(
	    labels_vector.vlen == num_vectors,
	    "The number of samples provided (%d)"
	    " must be equal to the number of labels provided(%d)\n",
	    num_vectors, labels_vector.vlen);

	// C holds the number of unique classes.
	int32_t C=((CMulticlassLabels*)labels)->get_num_classes();

	REQUIRE(C>1, "At least two classes are needed to perform LDA.\n")

	m_num_dim=num_dimensions;

	// clip number if Dimensions to be a valid number
	if ((m_num_dim<=0) || (m_num_dim>(C-1)))
		m_num_dim=(C-1);

	bool lda_more_efficient =
	    m_method == AUTO_FLDA && num_vectors < num_features;

	if ((m_method == CANVAR_FLDA) || lda_more_efficient)
		return solver_canvar(feature_matrix, labels_vector, C);
	else
		return solver_classic(feature_matrix, labels_vector, C);
}

void CFisherLDA::center_data_compute_means(
    SGMatrix<float64_t>& data, SGVector<float64_t>& labels_vector, int32_t C,
    vector<SGVector<float64_t>>& mean_class, std::vector<index_t>& num_class)
{
	// holds the total mean
	m_mean_vector = SGVector<float64_t>(data.num_rows);
	linalg::zero(m_mean_vector);
	for (index_t i = 0; i < C; ++i)
	{
		mean_class[i] = SGVector<float64_t>(data.num_rows);
		linalg::zero(mean_class[i]);
	}

	// calculate the class means and the total means.
	for (index_t i = 0; i < data.num_cols; i++)
	{
		index_t c = (index_t)labels_vector[i];
		num_class[c]++;
		linalg::add_col_vec(data, i, mean_class[c], mean_class[c]);
	}
	for (index_t i = 0; i < C; ++i)
	{
		linalg::add(m_mean_vector, mean_class[i], m_mean_vector);
		linalg::scale(
		    mean_class[i], mean_class[i], 1 / (float64_t)num_class[i]);
	}
	linalg::scale(m_mean_vector, m_mean_vector, 1 / (float64_t)data.num_cols);

	// Subtract the class means from the 'respective' data.
	// e.g all data belonging to class 0 is subtracted by
	// the mean of class 0 data.
	for (index_t i = 0; i < data.num_cols; i++)
		linalg::add_col_vec(
		    data, i, mean_class[labels_vector[i]], data, 1.0, -1.0);
}

bool CFisherLDA::solver_canvar(
    SGMatrix<float64_t>& data, SGVector<float64_t>& labels_vector, int32_t C)
{
	index_t num_vectors = data.num_cols;
	index_t num_features = data.num_rows;

	// holds the mean for each class
	vector<SGVector<float64_t>> mean_class(C);

	// holds the frequency for each class.
	// i.e the i'th element holds the number
	// of times class i is observed.
	vector<index_t> num_class(C);

	SGMatrix<float64_t> feature_centered = data.clone();
	center_data_compute_means(
	    feature_centered, labels_vector, C, mean_class, num_class);

	// holds the feature matrix for each class
	vector<SGMatrix<float64_t>> centered_class(C);
	vector<index_t> centered_class_col(C);

	SGMatrix<float64_t> Sw(num_features, num_features);
	linalg::zero(Sw);
	for (index_t i = 0; i < C; ++i)
	{
		centered_class[i] = SGMatrix<float64_t>(num_features, num_class[i]);
		linalg::zero(centered_class[i]);
	}
	for (index_t i = 0; i < num_vectors; i++)
	{
		index_t c = (index_t)labels_vector[i];
		auto vec = feature_centered.get_column(i);
		linalg::add_col_vec(
		    centered_class[c], centered_class_col[c], vec, centered_class[c]);
		centered_class_col[c]++;
	}
	for (index_t i = 0; i < C; ++i)
	{
		auto tmp = linalg::matrix_prod(
		    centered_class[i], centered_class[i], false, true);
		linalg::scale(tmp, tmp, num_class[i] / (float64_t)(num_class[i] - 1));
		linalg::add(Sw, tmp, Sw);
	}

	// regularization
	float64_t trace = linalg::trace(Sw);
	SGMatrix<float64_t> id(Sw.num_rows, Sw.num_rows);
	linalg::identity(id);
	linalg::add(Sw, id, Sw, (1.0 - m_gamma), trace * (m_gamma) / num_features);

	// within class matrix for canonical variates implementation
	SGMatrix<float64_t> Sb(num_features, C);
	linalg::zero(Sb);
	for (index_t i = 0; i < C; i++)
	{
		auto col = Sb.get_column(i);
		linalg::add(mean_class[i], m_mean_vector, col, 1.0, -1.0);
		linalg::scale(col, col, sqrt(num_class[i]));
	}

	index_t r = CMath::min(num_vectors, num_features);
	SGMatrix<float64_t> U(num_features, r);
	SGVector<float64_t> singularValues(r);
	linalg::svd(data, singularValues, U);
	// basis to represent the solution
	SGMatrix<float64_t> Q;

	if (num_features > num_vectors)
	{
		index_t j = 0;
		for (index_t i = 0; i < num_vectors; i++)
			if (singularValues[i] > m_threshold)
				j++;
			else
				break;
		Q = SGMatrix<float64_t>(U.matrix, num_features, j, false);
	}
	else
		Q = U;

	// Sb is the modified between scatter
	auto aux = linalg::matrix_prod(Q, Sb, true, false);
	Sb = linalg::matrix_prod(aux, aux, false, true);
	// Sw is the modified within scatter
	aux = linalg::matrix_prod(Q, Sw, true, false);
	Sw = linalg::matrix_prod(aux, Q);

	// To find svd(inverse(Sw)' * Sb * inverse(Sw))
	// solve Sb = (Sw' * X) * Sw
	// 1. get chol(Sw)
	// 2. solve chol(Sw)' * Y = Sb
	// 3. solve chol(Sw) * X = Y
	// 4. compute svd(X)
	auto chol = linalg::cholesky_factor(Sw);
	SGMatrix<float64_t> W(Sb.num_rows, Sb.num_cols);
	SGVector<float64_t> eigenvalues(Sb.num_rows);
	linalg::svd(
	    linalg::triangular_solver(
	        chol,
	        linalg::transpose_matrix(linalg::triangular_solver(chol, Sb))),
	    eigenvalues, W);

	m_transformation_matrix = SGMatrix<float64_t>(num_features, m_num_dim);
	linalg::zero(m_transformation_matrix);
	m_eigenvalues_vector = SGVector<float64_t>(m_num_dim);

	auto Wt = linalg::matrix_prod(Q, W);
	for (index_t i = 0; i < m_num_dim; ++i)
	{
		m_transformation_matrix.set_column(i, Wt.get_column(i));
		m_eigenvalues_vector[i] = eigenvalues[i];
	}

	return true;
}

bool CFisherLDA::solver_classic(
    SGMatrix<float64_t>& data, SGVector<float64_t>& labels_vector, int32_t C)
{
	index_t num_features = data.num_rows;

	// holds the mean for each class
	vector<SGVector<float64_t>> mean_class(C);

	// holds the frequency for each class.
	// i.e the i'th element holds the number
	// of times class i is observed.
	vector<index_t> num_class(C);

	SGMatrix<float64_t> feature_centered = data.clone();
	center_data_compute_means(
	    feature_centered, labels_vector, C, mean_class, num_class);

	// For holding the within class scatter.
	auto Sw =
	    linalg::matrix_prod(feature_centered, feature_centered, false, true);

	// For holding the between class scatter.
	SGMatrix<float64_t> Sb(num_features, C);

	for (index_t i = 0; i < C; i++)
		Sb.set_column(i, linalg::add(mean_class[i], m_mean_vector, 1.0, -1.0));
	Sb = linalg::matrix_prod(Sb, Sb, false, true);

	// solve Sw * M = Sb
	auto aux = linalg::qr_solver(Sw, Sb);

	// calculate the eigenvalues and eigenvectors of M.
	SGVector<float64_t> eigenvalues(Sb.num_rows);
	SGMatrix<float64_t> eigenvectors(Sb.num_rows, Sb.num_cols);
	linalg::eigen_solver(aux, eigenvalues, eigenvectors);

	// keep 'm_num_dim' numbers of top Eigenvalues
	m_eigenvalues_vector = SGVector<float64_t>(m_num_dim);

	// keep 'm_num_dim' numbers of EigenVectors
	// corresponding to their respective eigenvalues
	m_transformation_matrix = SGMatrix<float64_t>(num_features, m_num_dim);

	auto args = CMath::argsort(eigenvalues);
	for (index_t i = 0; i < m_num_dim; i++)
	{
		index_t k = args[num_features - i - 1];
		m_eigenvalues_vector[i] = eigenvalues[k];
		m_transformation_matrix.set_column(k, eigenvectors.get_column(i));
	}

	return true;
}

void CFisherLDA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
	m_mean_vector=SGVector<float64_t>();
	m_eigenvalues_vector=SGVector<float64_t>();
}

SGMatrix<float64_t> CFisherLDA::apply_to_feature_matrix(CFeatures*features)
{
	REQUIRE(features->get_feature_class()==C_DENSE,
			"LDA only works with dense features\n");

	REQUIRE(features->get_feature_type()==F_DREAL,
			"LDA only works with real features\n");

	SGMatrix<float64_t> m =
	    ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	int32_t num_vectors=m.num_cols;
	int32_t num_features=m.num_rows;

	SG_INFO("Transforming feature matrix\n")
	Map<MatrixXd> transform_matrix(
	    m_transformation_matrix.matrix, m_transformation_matrix.num_rows,
	    m_transformation_matrix.num_cols);

	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features)

	Map<MatrixXd> feature_matrix (m.matrix, num_features, num_vectors);

	feature_matrix.block(0, 0, m_num_dim, num_vectors) =
	    transform_matrix.transpose() * feature_matrix;

	SG_INFO("Form matrix of target dimension")
	for (int32_t col=0; col<num_vectors; col++)
	{
		for (int32_t row=0; row<m_num_dim; row++)
			m[col*m_num_dim+row]=feature_matrix(row, col);
	}
	m.num_rows=m_num_dim;
	m.num_cols=num_vectors;
	((CDenseFeatures<float64_t>*)features)->set_feature_matrix(m);
	return m;
}

SGVector<float64_t> CFisherLDA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SGVector<float64_t> result = SGVector<float64_t>(m_num_dim);
	Map<VectorXd> resultVec(result.vector, m_num_dim);
	Map<VectorXd> inputVec(vector.vector, vector.vlen);

	Map<VectorXd> mean(m_mean_vector.vector, m_mean_vector.vlen);
	Map<MatrixXd> transformMat(
	    m_transformation_matrix.matrix, m_transformation_matrix.num_rows,
	    m_transformation_matrix.num_cols);

	resultVec=transformMat.transpose()*inputVec;
	return result;
}

SGMatrix<float64_t> CFisherLDA::get_transformation_matrix()
{
	return m_transformation_matrix;
}

SGVector<float64_t> CFisherLDA::get_eigenvalues()
{
	return m_eigenvalues_vector;
}

SGVector<float64_t> CFisherLDA::get_mean()
{
	return m_mean_vector;
}
