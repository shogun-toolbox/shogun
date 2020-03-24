/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDSHOGUN_H_
#define SHOGUNADDSHOGUN_H_

#include <shogun/mathematics/graph/nodes/Add.h>
#include <shogun/mathematics/graph/ops/abstract/BinaryOperator.h>
#include <shogun/mathematics/graph/CPUArch.h>
#include <Eigen/Core>

namespace shogun
{
	namespace graph
	{
		namespace op
		{

			template <typename T>
			void add_kernel_implementation_avx512f(
			    void* input1, void* input2, void* output, const size_t size);

			template <typename T>
			void add_kernel_implementation_avx(
			    void* input1, void* input2, void* output);

			template <typename T>
			void add_kernel_implementation_sse2(
			    void* input1, void* input2, void* output, const size_t size);

			IGNORE_IN_CLASSLIST class AddShogun
			    : public ShogunBinaryOperator<AddShogun>
			{
			public:
				friend class ShogunBinaryOperator<AddShogun>;

				AddShogun(const std::shared_ptr<node::Node>& node)
				    : ShogunBinaryOperator(node)
				{
				}

				std::string_view get_operator_name() const final
				{
					return "Add";
				}

			protected:
				template <typename T>
				void kernel_implementation(
				    void* input1, void* input2, void* output, const size_t size)
				{
					auto* CPU_arch = CPUArch::instance();
					if (CPU_arch->has_avx512f())
						add_kernel_implementation_avx512f<T>(input1, input2, output, size);
					else if (CPU_arch->has_avx())
					{
						size_t i = 0;
						if (size > 32/sizeof(T))
						{
							size_t remainder = size % (32/sizeof(T));
							for (;i<size-remainder; i+=32/sizeof(T))
							{
								if constexpr(std::is_same_v<float, T>)
								{
									const auto packet1 = Eigen::internal::ploadu<Eigen::internal::Packet8f>(static_cast<const T*>(input1)+i);
									const auto packet2 = Eigen::internal::ploadu<Eigen::internal::Packet8f>(static_cast<const T*>(input2)+i);
									Eigen::internal::Packet8f output_packet;
									add_kernel_implementation_avx<T>((void*)&packet1, (void*)&packet2, (void*)&output_packet);
									Eigen::internal::pstoreu((static_cast<float*>(output)+i), output_packet);
								}
								else
								{
									add_kernel_implementation_avx<T>((void*)(static_cast<const T*>(input1)+i), 
										(void*)(static_cast<const T*>(input2)+i), 
										(void*)(static_cast<T*>(output)+i));
								}
							}
						}
						while(i < size)
						{
							*(static_cast<T*>(output)+i) = *(static_cast<const T*>(input1)+i) + *(static_cast<const T*>(input2)+i); 
							i++;
						}
					}
					else if (CPU_arch->has_sse2())
						add_kernel_implementation_sse2<T>(input1, input2, output, size);
					else
						std::transform(
						    static_cast<const T*>(input1),
						    static_cast<const T*>(input1) + size,
						    static_cast<const T*>(input2), static_cast<T*>(output),
						    std::plus<T>());
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
							    return *static_cast<const T*>(input1) + val;
						    });
					}
					else
					{
						std::transform(
						    static_cast<const T*>(input1),
						    static_cast<const T*>(input1) + size,
						    static_cast<T*>(output), [&input2](const T& val) {
							    return val + *static_cast<const T*>(input2);
						    });
					}
				}
			};
		} // namespace op
	}     // namespace graph
} // namespace shogun

#endif