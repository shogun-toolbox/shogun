#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/regression/GLM.h>
#include <utility>
using namespace shogun;

GLM::GLM(): LinearMachine(){
    init();
}

void GLM::init(){

}
GLM::GLM(DescendUpdaterWithCorrection* descend_updater, Family family, LinkFunction Link_fn, 
	float64_t alpha, float64_t lambda): LinearMachine(){
        m_alpha= alpha;
        m_lambda= lambda;
        m_linkfn= Link_fn;
        m_descend_updater = descend_updater;
        m_family= family;
    }