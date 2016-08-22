/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2016 Saurabh Mahindre
 */

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/multiclass/tree/RRCARTree.h>
#include <iostream>
using namespace Eigen;
using namespace shogun;

CRRCARTree::CRRCARTree(): CRandomCARTree()
{
	SG_ADD(&m_rotation_matrix, "m_rotation_matrix", "rotation matrix", MS_NOT_AVAILABLE);	
}

CRRCARTree::~CRRCARTree()
{
}

bool CRRCARTree::train_machine(CFeatures* data)
{
	REQUIRE(data,"Data required for training\n")
	REQUIRE(data->get_feature_class()==C_DENSE,"Dense data required for training\n")

	int32_t num_features=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_features();
	int32_t num_vectors=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_vectors();

	SGMatrix<float64_t> feats = (dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_feature_matrix();
	m_rotation_matrix = generate_rotation_matrix(num_features);

	Map<MatrixXd> map_feats (feats.matrix, feats.num_rows, feats.num_cols);
	Map<MatrixXd> map_rotation_matrix (m_rotation_matrix.matrix, num_features, num_features);

	SGMatrix<float64_t> new_mat = SGMatrix<float64_t>(num_features, num_vectors);
	new_mat.zero();
	Map<MatrixXd> map_new_mat(new_mat.matrix, num_features, num_vectors);
 
	MatrixXd temp = map_feats.transpose()*map_rotation_matrix;
	map_new_mat = temp.transpose();
	CFeatures* new_data = new CDenseFeatures<float64_t>(new_mat);
	return CRandomCARTree::train_machine(new_data);
}

SGMatrix<float64_t> CRRCARTree::generate_rotation_matrix(int32_t n)
{
	SGMatrix<float64_t> M(n, n);
	Map<MatrixXd> map_M(M.matrix, n, n);
	MatrixXd A(n, n);
	const VectorXd ones (VectorXd::Ones(n));
	
	for (int32_t i=0; i<n; ++i)
		for (int32_t j=0; j<n; ++j)
			A(i, j) = CMath::random(0.0, 1.0);
	
	const HouseholderQR<MatrixXd> qr(A);
	const MatrixXd Q = qr.householderQ();
	map_M = Q * (qr.matrixQR().diagonal().array() < 0).select(-ones, ones).asDiagonal();

	if (map_M.determinant() < 0)
		for (int32_t i =0; i<n ; ++i)
			map_M(i, 0) = -map_M(i, 0);	

	return M;
}

CMulticlassLabels* CRRCARTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	SGMatrix<float64_t> feats = (dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_feature_matrix();
	int32_t num_features=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_features();
	int32_t num_vectors=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_vectors();

	Map<MatrixXd> map_feats (feats.matrix, feats.num_rows, feats.num_cols);
	Map<MatrixXd> map_rotation_matrix (m_rotation_matrix.matrix, num_features, num_features);

	SGMatrix<float64_t> new_mat = SGMatrix<float64_t>(num_features, num_vectors);
	Map<MatrixXd> map_new_mat(new_mat.matrix, num_features, num_vectors);
 
	MatrixXd temp = map_feats.transpose()*map_rotation_matrix;
	map_new_mat = temp.transpose();
	CFeatures* new_data = new CDenseFeatures<float64_t>(new_mat);
	return CRandomCARTree::apply_multiclass(new_data);
}

CRegressionLabels* CRRCARTree::apply_regression(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	SGMatrix<float64_t> feats = (dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_feature_matrix();
	int32_t num_features=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_features();
	int32_t num_vectors=(dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_vectors();

	Map<MatrixXd> map_feats (feats.matrix, feats.num_rows, feats.num_cols);
	Map<MatrixXd> map_rotation_matrix (m_rotation_matrix.matrix, num_features, num_features);

	SGMatrix<float64_t> new_mat = SGMatrix<float64_t>(num_features, num_vectors);
	Map<MatrixXd> map_new_mat(new_mat.matrix, num_features, num_vectors);
 
	MatrixXd temp = map_feats.transpose()*map_rotation_matrix;
	map_new_mat = temp.transpose();
	CFeatures* new_data = new CDenseFeatures<float64_t>(new_mat);
	return CRandomCARTree::apply_regression(new_data);
}
