/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CLinearRidgeRegression::CLinearRidgeRegression()
: CLinearMachine()
{
	init();
}

CLinearRidgeRegression::CLinearRidgeRegression(float64_t tau, CDenseFeatures<float64_t>* data, CLabels* lab)
: CLinearMachine()
{
	init();

	m_tau=tau;
	set_labels(lab);
	set_features(data);
}

void CLinearRidgeRegression::init()
{
	m_tau=1e-6;

	SG_ADD(&m_tau, "tau", "Regularization parameter", MS_AVAILABLE);
}

bool CLinearRidgeRegression::train_machine(CFeatures* data)
{
    REQUIRE(m_labels,"No labels set\n");

    if (!data)
    	data=features;

    REQUIRE(data,"No features provided and no featured previously set\n");

    REQUIRE(m_labels->get_num_labels() == data->get_num_vectors(),
    	"Number of training vectors (%d) does not match number of labels (%d)\n",
    	m_labels->get_num_labels(), data->get_num_vectors());

    REQUIRE(data->get_feature_class() == C_DENSE,
    	"Expected Dense Features (%d) but got (%d)\n",
    	C_DENSE, data->get_feature_class());

    REQUIRE(data->get_feature_type() == F_DREAL,
    	"Expected Real Features (%d) but got (%d)\n",
    	F_DREAL, data->get_feature_type());

    CDenseFeatures<float64_t>* feats=(CDenseFeatures<float64_t>*) data;
    int32_t num_feat=feats->get_num_features();
    int32_t num_vec=feats->get_num_vectors();

    SGMatrix<float64_t> kernel_matrix(num_feat,num_feat);
    SGMatrix<float64_t> feats_matrix(feats->get_feature_matrix());
    SGVector<float64_t> y(num_feat);
    SGVector<float64_t> tau_vector(num_feat);

    tau_vector.zero();
    tau_vector.add(m_tau);

    Map<MatrixXd> eigen_kernel_matrix(kernel_matrix.matrix, num_feat,num_feat);
    Map<MatrixXd> eigen_feats_matrix(feats_matrix.matrix, num_feat,num_vec);
    Map<VectorXd> eigen_y(y.vector, num_feat);
    Map<VectorXd> eigen_labels(((CRegressionLabels*)m_labels)->get_labels(),num_vec);
    Map<VectorXd> eigen_tau(tau_vector.vector, num_feat);

    eigen_kernel_matrix = eigen_feats_matrix*eigen_feats_matrix.transpose();

    eigen_kernel_matrix.diagonal() += eigen_tau;

    eigen_y = eigen_feats_matrix*eigen_labels ;

    LLT<MatrixXd> llt;
    llt.compute(eigen_kernel_matrix);
    if(llt.info() != Eigen::Success)
    {
    	SG_WARNING("Features covariance matrix was not positive definite\n");
    	return false;
    }
    eigen_y = llt.solve(eigen_y);

    set_w(y);
    return true;
}

bool CLinearRidgeRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CLinearRidgeRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
#endif
