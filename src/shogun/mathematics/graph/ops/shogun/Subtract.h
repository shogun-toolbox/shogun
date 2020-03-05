// /*
//  * This software is distributed under BSD 3-clause license (see LICENSE
//  file).
//  *
//  * Authors: Gil Hoben
//  */

#ifndef SHOGUN_SUBTRACT_SHOGUN_H_
#define SHOGUN_SUBTRACT_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Subtract.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class SubtractShogun
			    : public ShogunBinaryOperator<SubtractShogun>
			{
			public:
				friend class ShogunBinaryOperator<SubtractShogun>;

				SubtractShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunBinaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Subtract";
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
					    std::minus<T>());
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif