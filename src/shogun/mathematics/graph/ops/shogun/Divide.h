/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNMULTIPLYSHOGUN_H_
#define SHOGUNMULTIPLYSHOGUN_H_

#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun {

	IGNORE_IN_CLASSLIST class DivideShogun : public BinaryOperator<DivideShogun, OperatorShogunBackend>
	{
	public:

		DivideShogun(): BinaryOperator() {};

		std::string_view get_operator_name() const override
		{
			return "Divide";
		}

		template <typename T>
		void kernel_implementation(
		    void* input1, void* input2, void* output, size_t size)
		{
			std::transform(
			    static_cast<const T*>(input1),
			    static_cast<const T*>(input1) + size,
			    static_cast<const T*>(input2), static_cast<T*>(output),
			    std::divides<T>());
		}
	};
}

#endif