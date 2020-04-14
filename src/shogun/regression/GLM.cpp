#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/regression/GLM.h>
#include <utility>
using namespace shogun;

GLM::GLM() : LinearMachine()
{
	init();
}
SGVector<float64_t> log_likelihood(
    const std::shared_ptr<DenseFeatures<float64_t>>& features,
    const std::shared_ptr<Labels>& label)
{
	auto vector_count = features->get_num_vectors();
	auto feature_count = features->get_num_features();
	ASSERT(vector_count > 0 && label->get_num_labels() == vector_count)
	// Array of Lambdas
	SGVector<float64_t> lambda(vector_count);
	for (auto i = 0; i < vector_count; i++)
	{
		SGVector<float64_t> feature_vector = features->get_feature_vector(i);
		// Assume beta is the same as the feature vector
		SGVector<float64_t> beta = feature_vector.clone();
		// Assume beta0 is the same as the first element in the feature vector
		float64_t b0 = feature_vector.get_element(0);
		float64_t res = linalg::dot(feature_vector, beta);
		lambda.set_element(log(1 + std::exp(b0 + res)), i);
	}
	SGVector<float64_t> likelihood(vector_count);
	SGVector<float64_t> labels = label->get_values();
	SGVector<float64_t> log_lambda(vector_count);

	for (auto i = 0; i < vector_count; i++)
		log_lambda.set_element(log(lambda.get_element(i)), i);

	likelihood = linalg::add(
	    linalg::element_prod(labels, log_lambda), lambda, 1.0, -1.0);
	SGVector<float64_t> likelihood_clone = likelihood.clone();
	for (auto i = 0; i < vector_count; i++)
		likelihood.set_element(
		    SGVector<float64_t>::sum(likelihood_clone, i), i);
	return likelihood;
}
void GLM::init()
{
	SG_ADD(
	    &m_alpha, "alpha", "Weighting parameter between L1 and L2 Penalty",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_lambda, "lambda", "Regularization parameter lambda",
	    ParameterProperties::HYPER);
	SG_ADD(
	    (std::shared_ptr<SGObject>*)&m_descend_updater, "descend_updater",
	    "Descend Updater used for updating weights",
	    ParameterProperties::SETTING);
	SG_ADD(
	    &m_family, "family", "Distribution Family used",
	    ParameterProperties::SETTING);
	SG_ADD(
	    &m_link_fn, "link_fn", "Link function used",
	    ParameterProperties::SETTING);
}
GLM::GLM(
    const std::shared_ptr<DescendUpdater>& descend_updater,
    DistributionFamily family, LinkFunction link_fn, float64_t alpha,
    float64_t lambda)
    : LinearMachine()
{
	m_alpha = alpha;
	m_lambda = lambda;
	m_link_fn = link_fn;
	m_descend_updater = descend_updater;
	m_family = family;
	init();
}