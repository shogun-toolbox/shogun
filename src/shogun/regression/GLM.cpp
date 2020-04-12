#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/regression/GLM.h>
#include <utility>
using namespace shogun;

GLM::GLM() : LinearMachine()
{
	init();
}

void GLM::init()
{
	SG_ADD(&m_alpha, "alpha", "alpha parameter", ParameterProperties::HYPER);
	SG_ADD(&m_lambda, "lambda", "lambda parameter", ParameterProperties::HYPER);
	SG_ADD(
	    (std::shared_ptr<SGObject>*)&m_descend_updater, "descend_updater",
	    "Descend Updater used for updating weights",
	    ParameterProperties::SETTING);
	SG_ADD(&m_family, "family", "family used", ParameterProperties::SETTING);
	SG_ADD(
	    &m_linkfn, "linkfn", "Link function used",
	    ParameterProperties::SETTING);
}
GLM::GLM(
    std::shared_ptr<DescendUpdater> descend_updater, Family family,
    LinkFunction Link_fn, float64_t alpha, float64_t lambda)
    : LinearMachine()
{
	m_alpha = alpha;
	m_lambda = lambda;
	m_linkfn = Link_fn;
	m_descend_updater = descend_updater;
	m_family = family;
}