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
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/preprocessor/FisherLDA.h>
#include <shogun/solver/LDACanVarSolver.h>
#include <shogun/solver/LDASolver.h>

using namespace std;
using namespace Eigen;
using namespace shogun;

FisherLDA::FisherLDA(
    int32_t num_dimensions, EFLDAMethod method, float64_t thresh,
    float64_t gamma, bool bdc_svd)
    : DensePreprocessor<float64_t>()
{
	initialize_parameters();
	m_num_dim = num_dimensions;
	m_method=method;
	m_threshold=thresh;
	m_gamma = gamma;
	m_bdc_svd = bdc_svd;
}

void FisherLDA::initialize_parameters()
{
	m_method=AUTO_FLDA;
	m_threshold=0.01;
	m_num_dim=0;
	m_gamma = 0;
	m_bdc_svd = true;
	SG_ADD(
	    &m_num_dim, "final_dimensions", "dimensions to be retained");
	SG_ADD(&m_gamma, "m_gamma", "Regularization parameter");
	SG_ADD(&m_bdc_svd, "m_bdc_svd", "Use BDC-SVD algorithm");
	SG_ADD(
	    &m_transformation_matrix, "transformation_matrix",
	    "Transformation"
	    " matrix (Eigenvectors of covariance matrix).");
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.");
	SG_ADD(
	    &m_eigenvalues_vector, "eigenvalues_vector", "Vector with Eigenvalues.");
	SG_ADD_OPTIONS(
	    &m_method, "method", "Matrix decomposition method.",
		ParameterProperties::NONE, SG_OPTIONS(AUTO_FLDA, CANVAR_FLDA,
			CLASSIC_FLDA));
}

FisherLDA::~FisherLDA()
{
}

void FisherLDA::fit(std::shared_ptr<Features> features)
{
	error("Labels for the given features are not specified!");
}

void FisherLDA::fit(std::shared_ptr<Features> features, std::shared_ptr<Labels> labels)
{
	require(features, "Features are not provided!");

	require(labels, "Labels for the given features are not specified!");

	require(
	    labels->get_label_type() == LT_MULTICLASS,
	    "The labels should be of "
	    "the type MulticlassLabels! you provided {}\n",
	    labels->get_name());

	auto dense_features = features->as<DenseFeatures<float64_t>>();
	auto mc =
	    multiclass_labels(labels);

	index_t num_vectors = dense_features->get_num_vectors();
	index_t num_features = dense_features->get_num_features();

	require(
	    labels->get_num_labels() == num_vectors,
	    "The number of samples provided ({})"
	    " must be equal to the number of labels provided({})\n",
	    num_vectors, labels->get_num_labels());

	int32_t num_class = mc->get_num_classes();

	require(num_class > 1, "At least two classes are needed to perform LDA.");

	// clip number if Dimensions to be a valid number
	if ((m_num_dim <= 0) || (m_num_dim > (num_class - 1)))
		m_num_dim = (num_class - 1);

	bool lda_more_efficient =
	    m_method == AUTO_FLDA && num_vectors < num_features;

	if ((m_method == CANVAR_FLDA) || lda_more_efficient)
		solver_canvar(dense_features, mc);
	else
		solver_classic(dense_features, mc);
}

void FisherLDA::solver_canvar(
    std::shared_ptr<DenseFeatures<float64_t>> features, std::shared_ptr<MulticlassLabels> labels)
{
	auto solver = std::unique_ptr<LDACanVarSolver<float64_t>>(
	    new LDACanVarSolver<float64_t>(
	        features, labels, m_num_dim, m_gamma, m_bdc_svd, m_threshold));

	m_transformation_matrix = solver->get_eigenvectors();
	m_eigenvalues_vector = solver->get_eigenvalues();
}

void FisherLDA::solver_classic(
    std::shared_ptr<DenseFeatures<float64_t>> features, std::shared_ptr<MulticlassLabels> labels)
{
	SGMatrix<float64_t> data = features->get_feature_matrix();
	index_t num_features = data.num_rows;
	int32_t num_class = labels->get_num_classes();

	auto solver = std::unique_ptr<LDASolver<float64_t>>(
	    new LDASolver<float64_t>(features, labels, m_gamma));

	m_mean_vector = solver->get_mean();
	auto class_mean = solver->get_class_mean();
	auto class_count = solver->get_class_count();
	SGMatrix<float64_t> Sw = solver->get_within_cov();

	// For holding the between class scatter.
	SGMatrix<float64_t> Sb(num_features, num_class);

	for (index_t i = 0; i < num_class; i++)
		Sb.set_column(i, linalg::add(class_mean[i], m_mean_vector, 1.0, -1.0));
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

	auto args = Math::argsort(eigenvalues);
	for (index_t i = 0; i < m_num_dim; i++)
	{
		index_t k = args[num_features - i - 1];
		m_eigenvalues_vector[i] = eigenvalues[k];
		m_transformation_matrix.set_column(k, eigenvectors.get_column(i));
	}
}

void FisherLDA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
	m_mean_vector=SGVector<float64_t>();
	m_eigenvalues_vector=SGVector<float64_t>();
}

SGMatrix<float64_t> FisherLDA::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	auto num_vectors = matrix.num_cols;
	auto num_features = matrix.num_rows;

	io::info("Transforming feature matrix");
	Map<MatrixXd> transform_matrix(
	    m_transformation_matrix.matrix, m_transformation_matrix.num_rows,
	    m_transformation_matrix.num_cols);

	io::info("get Feature matrix: {}x{}", num_vectors, num_features);

	Map<MatrixXd> feature_matrix(matrix.matrix, num_features, num_vectors);

	feature_matrix.block(0, 0, m_num_dim, num_vectors) =
	    transform_matrix.transpose() * feature_matrix;

	io::info("Form matrix of target dimension");
	for (int32_t col=0; col<num_vectors; col++)
	{
		for (int32_t row=0; row<m_num_dim; row++)
			matrix[col * m_num_dim + row] = feature_matrix(row, col);
	}
	matrix.num_rows = m_num_dim;
	matrix.num_cols = num_vectors;
	return matrix;
}

SGVector<float64_t> FisherLDA::apply_to_feature_vector(SGVector<float64_t> vector)
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

SGMatrix<float64_t> FisherLDA::get_transformation_matrix()
{
	return m_transformation_matrix;
}

SGVector<float64_t> FisherLDA::get_eigenvalues()
{
	return m_eigenvalues_vector;
}

SGVector<float64_t> FisherLDA::get_mean()
{
	return m_mean_vector;
}
