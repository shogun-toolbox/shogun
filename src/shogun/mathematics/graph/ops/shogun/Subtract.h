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
#include <shogun/mathematics/graph/CPUArch.h>

namespace shogun
{
	namespace graph
	{
		namespace op
		{

			template <typename T>
			void subtract_kernel_implementation_avx512f(
			    void* input1, void* input2, void* output, const size_t size);

			template <typename T>
			void subtract_kernel_implementation_avx(
			    void* input1, void* input2, void* output, const size_t size);

			template <typename T>
			void subtract_kernel_implementation_sse2(
			    void* input1, void* input2, void* output, const size_t size);

			// uses _mm_min_epi32
			template <typename T>
			void subtract_kernel_implementation_sse41(
			    void* input1, void* input2, void* output, const size_t size);

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
				    void* input1, void* input2, void* output, const size_t size)
				{
					auto* CPU_arch = CPUArch::instance();
					if (CPU_arch->has_avx512f())
						subtract_kernel_implementation_avx512f<T>(input1, input2, output, size);
					else if (CPU_arch->has_avx())
						subtract_kernel_implementation_avx<T>(input1, input2, output, size);
					else if (CPU_arch->has_sse4_1() && std::is_same_v<int32_t, T>)
					{
						// only lookup for int32 implementation at compile time
						if constexpr(std::is_same_v<int32_t, T>)
						{
							// _mm_min_epi32
							subtract_kernel_implementation_sse41<T>(input1, input2, output, size);
						}
					}
					else if (CPU_arch->has_sse2())
						subtract_kernel_implementation_sse2<T>(input1, input2, output, size);
					else
						std::transform(
						    static_cast<const T*>(input1),
						    static_cast<const T*>(input1) + size,
						    static_cast<const T*>(input2), static_cast<T*>(output),
						    std::minus<T>());
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
							    return *static_cast<const T*>(input1) - val;
						    });
					}
					else
					{
						std::transform(
						    static_cast<const T*>(input1),
						    static_cast<const T*>(input1) + size,
						    static_cast<T*>(output), [&input2](const T& val) {
							    return val - *static_cast<const T*>(input2);
						    });
					}
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif