/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Chiyuan Zhang, Viktor Gal, 
 *          Abhinav Rai, Youssef Emad El-Din
 */
#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/regression/LinearRidgeRegression.h>

using namespace shogun;

CLinearRidgeRegression::CLinearRidgeRegression()
: CLinearMachine()
{
	init();
}

CLinearRidgeRegression::CLinearRidgeRegression(float64_t tau, CDenseFeatures<float64_t>* data, CLabels* lab)
: CLinearMachine()
{
	init();

	set_tau(tau);
	set_labels(lab);
	set_features(data);
}

void CLinearRidgeRegression::init()
{
	set_tau(1e-6);

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

    SGMatrix<float64_t> kernel_matrix(num_feat,num_feat);
    SGMatrix<float64_t> feats_matrix(feats->get_feature_matrix());
    SGVector<float64_t> y(num_feat);
    SGVector<float64_t> tau_vector(num_feat);

    tau_vector.zero();
    tau_vector.add(m_tau);

	linalg::matrix_prod(feats_matrix, feats_matrix, kernel_matrix, false, true);
	linalg::add_diag(kernel_matrix, tau_vector);

	auto labels = regression_labels(m_labels);
	linalg::matrix_prod(feats_matrix, labels->get_labels(), y);

	auto decomposition = linalg::cholesky_factor(kernel_matrix);
	y = linalg::cholesky_solver(decomposition, y);
	auto lab = regression_labels(m_labels)->get_labels();
	auto intercept = linalg::mean(y) - linalg::dot(y, feats->mean<float64_t>());
	set_bias(intercept);

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
