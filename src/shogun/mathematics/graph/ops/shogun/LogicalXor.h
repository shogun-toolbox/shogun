/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_LOGICAL_XOR_SHOGUN_H_
#define SHOGUN_LOGICAL_XOR_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/LogicalXor.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class LogicalXorShogun
			    : public ShogunBinaryOperator<LogicalXorShogun>
			{
			public:
				friend class ShogunBinaryOperator<LogicalXorShogun>;

				LogicalXorShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunBinaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "LogicalXor";
				}

			protected:
				template <typename T>
				void kernel_implementation(
				    void* input1, void* input2, void* output, size_t size)
				{
					std::transform(
					    static_cast<const T*>(input1),
					    static_cast<const T*>(input1) + size,
					    static_cast<const T*>(input2), static_cast<T*>(output),
					    [](const T& lhs, const T& rhs) {
						    return !lhs != !rhs;
					    });
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif