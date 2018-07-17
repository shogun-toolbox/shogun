/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn, 
 *          Michele Mazzoni, Bjoern Esser, Fernando Iglesias, Abhijeet Kislay, 
 *          Viktor Gal, Evan Shelhamer, Giovanni De Toni, 
 *          Christopher Goldsworthy
 */
#include <shogun/lib/config.h>

#include <shogun/classifier/LDA.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/FisherLDA.h>
#include <shogun/solver/LDACanVarSolver.h>
#include <shogun/solver/LDASolver.h>
#include <vector>

using namespace Eigen;
using namespace shogun;

CLDA::CLDA(float64_t gamma, ELDAMethod method, bool bdc_svd)
    : CDenseFeaturesDispatch<CLDA, CLinearMachine>()
{
	init();
	m_method=method;
	m_gamma=gamma;
	m_bdc_svd = bdc_svd;
}

CLDA::CLDA(
    float64_t gamma, CDenseFeatures<float64_t>* traindat, CLabels* trainlab,
    ELDAMethod method, bool bdc_svd)
    : CDenseFeaturesDispatch<CLDA, CLinearMachine>(), m_gamma(gamma)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
	m_method=method;
	m_gamma=gamma;
	m_bdc_svd = bdc_svd;
}

void CLDA::init()
{
	m_method=AUTO_LDA;
	m_gamma=0;
	m_bdc_svd = true;
	set_compute_bias(false);

	SG_ADD(
	    (machine_int_t*)&m_method, "m_method",
	    "Method used for LDA calculation", MS_NOT_AVAILABLE);
	SG_ADD(&m_gamma, "m_gamma", "Regularization parameter", MS_AVAILABLE);
	SG_ADD(&m_bdc_svd, "m_bdc_svd", "Use BDC-SVD algorithm", MS_NOT_AVAILABLE);
}

CLDA::~CLDA()
{
}

template <typename ST>
bool CLDA::train_machine_templated(CDenseFeatures<ST>* data)
{
	index_t num_feat = data->get_num_features();
	index_t num_vec = data->get_num_vectors();
	;

	bool lda_more_efficient = (m_method == AUTO_LDA && num_vec <= num_feat);

	if (m_method == SVD_LDA || lda_more_efficient)
		return solver_svd<ST>(data);
	else
		return solver_classic<ST>(data);
}

template <typename ST>
bool CLDA::solver_svd(CDenseFeatures<ST>* data)
{
	auto labels = multiclass_labels(m_labels);
	REQUIRE(
	    labels->get_num_classes() == 2, "Number of classes (%d) must be 2\n",
	    labels->get_num_classes())

	// keep just one dimension to do binary classification
	const index_t projection_dim = 1;
	auto solver = std::unique_ptr<LDACanVarSolver<ST>>(
	    new LDACanVarSolver<ST>(
	        data, labels, projection_dim, m_gamma, m_bdc_svd));

	SGVector<ST> w_st(solver->get_eigenvectors());

	auto class_mean = solver->get_class_mean();
	ST m_neg = linalg::dot(w_st, class_mean[0]);
	ST m_pos = linalg::dot(w_st, class_mean[1]);

	// change the sign of w if needed to get the correct labels
	float64_t sign = (m_pos > m_neg) ? 1 : -1;

	SGVector<float64_t> w(data->get_num_features());
	// copy w_st into w
	for (index_t i = 0; i < w.size(); ++i)
		w[i] = sign * w_st[i];
	set_w(w);

	set_bias(-0.5 * sign * (m_neg + m_pos));

	return true;
}

template <typename ST>
bool CLDA::solver_classic(CDenseFeatures<ST>* data)
{
	auto labels = multiclass_labels(m_labels);
	REQUIRE(
	    labels->get_num_classes() == 2, "Number of classes (%d) must be 2\n",
	    labels->get_num_classes())
	index_t num_feat = data->get_num_features();

	auto solver = std::unique_ptr<LDASolver<ST>>(
	    new LDASolver<ST>(data, labels, m_gamma));

	auto class_mean = solver->get_class_mean();
	auto class_count = solver->get_class_count();
	SGMatrix<ST> scatter_matrix = solver->get_within_cov();

	// the usual way
	// we need to find a Basic Linear Solution of A.x=b for 'x'.
	// Instead of crudely Inverting A, we go for solve() using Decompositions.
	// where:
	// MatrixXd A=scatter;
	// VectorXd b=mean_pos-mean_neg;
	// VectorXd x=w;
	auto decomposition = linalg::cholesky_factor(scatter_matrix);
	SGVector<ST> w_st = linalg::cholesky_solver(
	    decomposition,
	    linalg::add(class_mean[1], class_mean[0], (ST)1, (ST)-1));

	// get the weights w_neg(for -ve class) and w_pos(for +ve class)
	auto w_neg = linalg::cholesky_solver(decomposition, class_mean[0]);
	auto w_pos = linalg::cholesky_solver(decomposition, class_mean[1]);

	SGVector<float64_t> w(num_feat);
	// copy w_st into w
	for (index_t i = 0; i < w.size(); ++i)
		w[i] = (float64_t)w_st[i];
	set_w(w);

	// get the bias.
	set_bias(
	    (float64_t)(
	        0.5 * (linalg::dot(w_neg, class_mean[0]) -
	               linalg::dot(w_pos, class_mean[1]))));

	return true;
}
