/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DIVIDE_SHOGUN_H_
#define SHOGUN_DIVIDE_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Divide.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun
{
	namespace graph 
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class DivideShogun : public ShogunBinaryOperator<DivideShogun>
			{
			public:
				friend class ShogunBinaryOperator<DivideShogun>;
				
				DivideShogun(const std::shared_ptr<node::Node>& node) : ShogunBinaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Divide";
				}

			protected:
				template <typename T>
				void kernel_implementation(
				    void* input1, void* input2, void* output, size_t size)
				{
					// if we have SYCL or MSVC we could add parallel execution
					// or just use Eigen here
					std::transform(
					    static_cast<const T*>(input1),
					    static_cast<const T*>(input1) + size,
					    static_cast<const T*>(input2), static_cast<T*>(output),
					    std::divides<T>());
				}
			};
		}
	}
} // namespace shogun

#endif