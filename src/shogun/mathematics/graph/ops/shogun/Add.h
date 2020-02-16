/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDSHOGUN_H_
#define SHOGUNADDSHOGUN_H_

#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>

namespace shogun {

	IGNORE_IN_CLASSLIST class AddShogun : public ShogunBinaryOperator<AddShogun, Add>
	{
	public:

		friend class ShogunBinaryOperator<AddShogun, Add>;

		AddShogun(): ShogunBinaryOperator() {}

		std::string_view get_operator_name() const final
		{
			return "Add";
		}

		void build_implementation() final
		{
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
			    std::plus<T>());
		}
	};
}

#endif