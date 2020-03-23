/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUN_DIVIDE_SHOGUN_H_
#define SHOGUN_DIVIDE_SHOGUN_H_

#include <shogun/mathematics/graph/nodes/Divide.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>
#include <shogun/mathematics/graph/CPUArch.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			template <typename T>
			void divide_kernel_implementation_avx512f(
			    void* input1, void* input2, void* output, const size_t size);

			template <typename T>
			void divide_kernel_implementation_avx(
			    void* input1, void* input2, void* output, const size_t size);

			template <typename T>
			void divide_kernel_implementation_sse2(
			    void* input1, void* input2, void* output, const size_t size);

			IGNORE_IN_CLASSLIST class DivideShogun
			    : public ShogunBinaryOperator<DivideShogun>
			{
			public:
				friend class ShogunBinaryOperator<DivideShogun>;

				DivideShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunBinaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Divide";
				}

			protected:
				template <typename T>
				void kernel_implementation(
				    void* input1, void* input2, void* output, const size_t size)
				{
					auto* CPU_arch = CPUArch::instance();
					if (CPU_arch->has_avx512f())
						divide_kernel_implementation_avx512f<T>(input1, input2, output, size);
					else if (CPU_arch->has_avx())
						divide_kernel_implementation_avx<T>(input1, input2, output, size);
					else if (CPU_arch->has_sse2())
						divide_kernel_implementation_sse2<T>(input1, input2, output, size);
					else
						std::transform(
						    static_cast<const T*>(input1),
						    static_cast<const T*>(input1) + size,
						    static_cast<const T*>(input2), static_cast<T*>(output),
						    std::divides<T>());
				}

				template <typename T>
				void kernel_scalar_implementation(
				    void* input1, void* input2, void* output, const size_t size,
				    const bool scalar_first)
				{
					if (scalar_first)
					{
						std::transform(
						    static_cast<const T*>(input2),
						    static_cast<const T*>(input2) + size,
						    static_cast<T*>(output), [&input1](const T& val) {
							    return *static_cast<const T*>(input1) / val;
						    });
					}
					else
					{
						std::transform(
						    static_cast<const T*>(input1),
						    static_cast<const T*>(input1) + size,
						    static_cast<T*>(output), [&input2](const T& val) {
							    return val / *static_cast<const T*>(input2);
						    });
					}
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif