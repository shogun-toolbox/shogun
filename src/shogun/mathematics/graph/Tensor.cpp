#include <shogun/mathematics/graph/Tensor.h>

#include <shogun/io/fmt/fmt.h>

using namespace shogun;

std::string Tensor::to_string() const
{
	return fmt::format("Tensor(shape={}, type={})", m_shape, m_type);
}