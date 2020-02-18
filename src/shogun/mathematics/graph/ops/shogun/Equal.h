/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_EQUAL_SHOGUN_H_
#define SHOGUN_EQUAL_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Equal.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class EqualShogun
			    : public ShogunBinaryOperator<EqualShogun>
			{
			public:
				friend class ShogunBinaryOperator<EqualShogun>;

				EqualShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunBinaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Equal";
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
					    std::equal_to<T>());
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif