/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNMULTIPLYSHOGUN_H_
#define SHOGUNMULTIPLYSHOGUN_H_

#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun {

	IGNORE_IN_CLASSLIST class MultiplyShogun : public ShogunBinaryOperator<MultiplyShogun>
	{
	public:
		friend class ShogunBinaryOperator<MultiplyShogun>;

		MultiplyShogun(): ShogunBinaryOperator() {};

		std::string_view get_operator_name() const final
		{
			return "Multiply";
		}

		void build_implementation() final
		{
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
			    std::multiplies<T>());
		}
	};
}

#endif