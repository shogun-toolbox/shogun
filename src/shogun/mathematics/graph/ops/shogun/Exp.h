/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_EXP_SHOGUN_H_
#define SHOGUN_EXP_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Exp.h>
#include <shogun/mathematics/graph/ops/abstract/UnaryOperator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class ExpShogun
			    : public ShogunUnaryOperator<ExpShogun>
			{
			public:
				friend class ShogunUnaryOperator<ExpShogun>;

				ExpShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunUnaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Exp";
				}

			protected:
				template <typename T>
				void
				kernel_implementation(void* input, void* output, size_t size)
				{
					std::transform(
					    static_cast<const T*>(input),
					    static_cast<const T*>(input) + size,
					    static_cast<T*>(output),
					    [](const T& el) { return std::exp(el); });
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif