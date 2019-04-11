/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Evan Shelhamer, Thoralf Klein, Soeren Sonnenburg,
 *          Chiyuan Zhang
 */

#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;

namespace shogun
{
CEmbeddingConverter::CEmbeddingConverter()
: CConverter()
{
	m_target_dim = 1;
	m_distance = new CEuclideanDistance();
	SG_REF(m_distance);
	m_kernel = new CLinearKernel();
	SG_REF(m_kernel);

	init();
}

CEmbeddingConverter::~CEmbeddingConverter()
{
	SG_UNREF(m_distance);
	SG_UNREF(m_kernel);
}

void CEmbeddingConverter::set_target_dim(int32_t dim)
{
	ASSERT(dim>0)
	m_target_dim = dim;
}

int32_t CEmbeddingConverter::get_target_dim() const
{
	return m_target_dim;
}

void CEmbeddingConverter::set_distance(CDistance* distance)
{
	SG_REF(distance);
	SG_UNREF(m_distance);
	m_distance = distance;
}

CDistance* CEmbeddingConverter::get_distance() const
{
	SG_REF(m_distance);
	return m_distance;
}

void CEmbeddingConverter::set_kernel(CKernel* kernel)
{
	SG_REF(kernel);
	SG_UNREF(m_kernel);
	m_kernel = kernel;
}

CKernel* CEmbeddingConverter::get_kernel() const
{
	SG_REF(m_kernel);
	return m_kernel;
}

void CEmbeddingConverter::init()
{
	SG_ADD(&m_target_dim, "target_dim",
      "target dimensionality of preprocessor", ParameterProperties::HYPER);
	SG_ADD(
		&m_distance, "distance", "distance to be used for embedding",
		ParameterProperties::HYPER);
	SG_ADD(
		&m_kernel, "kernel", "kernel to be used for embedding", ParameterProperties::HYPER);
}
}
