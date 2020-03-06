/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_SIGN_SHOGUN_H_
#define SHOGUN_SIGN_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Sign.h>
#include <shogun/mathematics/graph/ops/abstract/UnaryOperator.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			IGNORE_IN_CLASSLIST class SignShogun
			    : public ShogunUnaryOperator<SignShogun>
			{
			public:
				friend class ShogunUnaryOperator<SignShogun>;

				SignShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunUnaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Sign";
				}

			protected:
				template <typename T>
				void
				kernel_implementation(void* input, void* output, size_t size)
				{
					if constexpr (!std::is_unsigned_v<T>)
					{
						std::transform(
						    static_cast<const T*>(input),
						    static_cast<const T*>(input) + size,
						    static_cast<T*>(output), [](const T& el) -> T {
							    if (el < T{0})
								    return T{-1};
							    return el > T{0};
						    });
					}
					else
					{
						error("Unsigned types not supported in Sign op.");
					}
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif