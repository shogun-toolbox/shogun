#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/distance/EuclidianDistance.h>

using namespace shogun;

CDimensionReductionPreprocessor::CDimensionReductionPreprocessor()
: CSimplePreprocessor<float64_t>()
{
	m_target_dim = 1;
	m_distance = new CEuclidianDistance();
	m_distance->parallel = this->parallel;
	SG_REF(this->parallel);
	m_kernel = new CGaussianKernel();
	m_kernel->parallel = this->parallel;
	SG_REF(this->parallel);

	init();
}

CDimensionReductionPreprocessor::~CDimensionReductionPreprocessor() 
{
	delete m_distance;
	SG_UNREF(this->parallel);
	delete m_kernel;
	SG_UNREF(this->parallel);
}

bool CDimensionReductionPreprocessor::init(CFeatures* data)
{
	return true;
}

void CDimensionReductionPreprocessor::cleanup()
{

}

SGMatrix<float64_t> CDimensionReductionPreprocessor::apply_to_feature_matrix(CFeatures* features)
{
	return ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
};

SGVector<float64_t> CDimensionReductionPreprocessor::apply_to_feature_vector(SGVector<float64_t> vector)
{
	return vector;
};

EPreprocessorType CDimensionReductionPreprocessor::get_type() const { return P_DIMENSIONREDUCTIONPREPROCESSOR; };

void CDimensionReductionPreprocessor::set_target_dim(int32_t dim)
{
	ASSERT(dim>0 || dim==AUTO_TARGET_DIM);
	m_target_dim = dim;
}

int32_t CDimensionReductionPreprocessor::get_target_dim() const
{
	return m_target_dim;
}

void CDimensionReductionPreprocessor::set_distance(CDistance* distance)
{
	SG_UNREF(m_distance);
	SG_REF(distance);
	m_distance = distance;
	m_distance->parallel = this->parallel;
	SG_REF(this->parallel);
}

CDistance* CDimensionReductionPreprocessor::get_distance() const
{
	SG_REF(m_distance);
	return m_distance;
}

void CDimensionReductionPreprocessor::set_kernel(CKernel* kernel)
{
	SG_UNREF(m_kernel);
	SG_REF(kernel);
	m_kernel = kernel;
	m_kernel->parallel = this->parallel;
	SG_REF(this->parallel);
}

CKernel* CDimensionReductionPreprocessor::get_kernel() const
{
	SG_REF(m_kernel);
	return m_kernel;
}

int32_t CDimensionReductionPreprocessor::detect_dim(SGMatrix<float64_t> distance_matrix)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

void CDimensionReductionPreprocessor::init()
{
	m_parameters->add(&m_target_dim, "target_dim",
					  "target dimensionality of preprocessor");
	m_parameters->add((CSGObject**)&m_distance, "distance",
					  "distance to be used for embedding");
	m_parameters->add((CSGObject**)&m_kernel, "kernel",
					  "kernel to be used for embedding");
}
