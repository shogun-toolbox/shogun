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
EmbeddingConverter::EmbeddingConverter()
: Converter()
{
	m_target_dim = 1;
	m_distance = std::make_shared<EuclideanDistance>();
	
	m_kernel = std::make_shared<LinearKernel>();
	

	init();
}

EmbeddingConverter::~EmbeddingConverter()
{
	
	
}

void EmbeddingConverter::set_target_dim(int32_t dim)
{
	ASSERT(dim>0)
	m_target_dim = dim;
}

int32_t EmbeddingConverter::get_target_dim() const
{
	return m_target_dim;
}

void EmbeddingConverter::set_distance(std::shared_ptr<Distance> distance)
{
	
	
	m_distance = distance;
}

std::shared_ptr<Distance> EmbeddingConverter::get_distance() const
{
	
	return m_distance;
}

void EmbeddingConverter::set_kernel(std::shared_ptr<Kernel> kernel)
{
	
	
	m_kernel = kernel;
}

std::shared_ptr<Kernel> EmbeddingConverter::get_kernel() const
{
	
	return m_kernel;
}

void EmbeddingConverter::init()
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
