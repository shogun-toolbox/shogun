#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/distance/EuclidianDistance.h>

using namespace shogun;

namespace shogun 
{
template<class ST>
CDimensionReductionPreprocessor<ST>::CDimensionReductionPreprocessor()
: CSimplePreprocessor<ST>()
{
	m_target_dim = 1;
	m_distance = new CEuclidianDistance();
	m_kernel = new CLinearKernel();

	init();
}

template<class ST>
CDimensionReductionPreprocessor<ST>::~CDimensionReductionPreprocessor() 
{
	SG_UNREF(m_distance);
	SG_UNREF(m_kernel);
}

template <class ST>
bool CDimensionReductionPreprocessor<ST>::init(CFeatures* data)
{
	return true;
}

template <class ST>
void CDimensionReductionPreprocessor<ST>::cleanup()
{

}

template<class ST>
EPreprocessorType CDimensionReductionPreprocessor<ST>::get_type() const { return P_DIMENSIONREDUCTIONPREPROCESSOR; };

template<class ST>
void CDimensionReductionPreprocessor<ST>::set_target_dim(int32_t dim)
{
	m_target_dim = dim;
}

template<class ST>
int32_t CDimensionReductionPreprocessor<ST>::get_target_dim() const
{
	return m_target_dim;
}

template<class ST>
int32_t CDimensionReductionPreprocessor<ST>::calculate_effective_target_dim(int32_t dim)
{
	if (m_target_dim<0)
	{
		if (dim+m_target_dim>0)
		{
			return dim+m_target_dim;
		}
		else
			return -1;
	}
	else
		return m_target_dim;
}

template<class ST>
void CDimensionReductionPreprocessor<ST>::set_distance(CDistance* distance)
{
	SG_UNREF(m_distance);
	SG_REF(distance);
	m_distance = distance;
}

template<class ST>
CDistance* CDimensionReductionPreprocessor<ST>::get_distance() const
{
	SG_REF(m_distance);
	return m_distance;
}

template<class ST>
void CDimensionReductionPreprocessor<ST>::set_kernel(CKernel* kernel)
{
	SG_UNREF(m_kernel);
	SG_REF(kernel);
	m_kernel = kernel;
}

template<class ST>
CKernel* CDimensionReductionPreprocessor<ST>::get_kernel() const
{
	SG_REF(m_kernel);
	return m_kernel;
}

template<class ST>
void CDimensionReductionPreprocessor<ST>::init()
{
	this->m_parameters->add(&m_target_dim, "target_dim",
					  "target dimensionality of preprocessor");
	this->m_parameters->add((CSGObject**)&m_distance, "distance",
					  "distance to be used for embedding");
	this->m_parameters->add((CSGObject**)&m_kernel, "kernel",
					  "kernel to be used for embedding");
}

template class CDimensionReductionPreprocessor<float32_t>;
template class CDimensionReductionPreprocessor<float64_t>;
}
