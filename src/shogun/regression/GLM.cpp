#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/regression/GLM.h>
#include <utility>
using namespace shogun;

GLM::GLM() : LinearMachine()
{
	init();
}
SGVector<float64_t>
log_likelihood(SGVector<float64_t> features, float64_t label)
{
	ASSERT(features.size() > 0)
	// Assume weights is the same as the feature vector
	SGVector<float64_t> weights = SGMatrix(features.clone());
	// Assume b0 to be the element in the beginning of the feature vector
	float64_t b0 = features.get_element(0);
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