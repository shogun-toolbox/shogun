/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/mathematics/graph/Tensor.h>

#include <shogun/io/fmt/fmt.h>

using namespace shogun;
using namespace shogun::graph;

std::string Tensor::to_string() const
{
	return fmt::format("Tensor(shape={}, type={})", m_shape, m_type);
}

template Tensor::Tensor(const SGVector<float32_t>&);
template Tensor::Tensor(const SGVector<float64_t>&);
template Tensor::Tensor(const SGMatrix<float32_t>&);
template Tensor::Tensor(const SGMatrix<float64_t>&);