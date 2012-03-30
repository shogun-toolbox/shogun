#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/distance/EuclidianDistance.h>

using namespace shogun;

namespace shogun
{
CDimensionReductionPreprocessor::CDimensionReductionPreprocessor()
: CSimplePreprocessor<float64_t>()
{
	m_target_dim = 1;
	m_distance = new CEuclidianDistance();
	m_kernel = new CLinearKernel();
	m_converter = NULL;

	init();
}

CDimensionReductionPreprocessor::CDimensionReductionPreprocessor(CEmbeddingConverter* converter)
: CSimplePreprocessor<float64_t>()
{
	SG_REF(converter);
	m_target_dim = 1;
	m_distance = new CEuclidianDistance();
	m_kernel = new CLinearKernel();
	m_converter = converter;

	init();
}

CDimensionReductionPreprocessor::~CDimensionReductionPreprocessor()
{
	SG_UNREF(m_distance);
	SG_UNREF(m_kernel);
	SG_UNREF(m_converter);
}

SGMatrix<float64_t> CDimensionReductionPreprocessor::apply_to_feature_matrix(CFeatures* features)
{
	if (m_converter)
	{
		m_converter->set_target_dim(m_target_dim);
		CSimpleFeatures<float64_t>* embedding = m_converter->embed(features);
		SGMatrix<float64_t> embedding_feature_matrix = embedding->steal_feature_matrix();
		((CSimpleFeatures<float64_t>*)features)->set_feature_matrix(embedding_feature_matrix);
		delete embedding;
		return embedding_feature_matrix;
	}
	else
	{
		SG_WARNING("Converter to process was not set.\n");
		return ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
	}
}

bool CDimensionReductionPreprocessor::init(CFeatures* data)
{
	return true;
}

void CDimensionReductionPreprocessor::cleanup()
{

}

EPreprocessorType CDimensionReductionPreprocessor::get_type() const { return P_DIMENSIONREDUCTIONPREPROCESSOR; };

void CDimensionReductionPreprocessor::set_target_dim(int32_t dim)
{
	ASSERT(dim>0);
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
}

CKernel* CDimensionReductionPreprocessor::get_kernel() const
{
	SG_REF(m_kernel);
	return m_kernel;
}

void CDimensionReductionPreprocessor::init()
{
	m_parameters->add((CSGObject**)&m_converter, "converter",
					  "embedding converter used to apply to data");
	SG_ADD(&m_target_dim, "target_dim",
					  "target dimensionality of preprocessor", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_distance, "distance",
					  "distance to be used for embedding", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_kernel, "kernel",
					  "kernel to be used for embedding", MS_AVAILABLE);
}
}
