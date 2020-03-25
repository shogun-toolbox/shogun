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
#include "Packet.h"


namespace shogun
{
	namespace graph
	{
		namespace op
		{
			template <typename T>
			void add_kernel_implementation_avx512f(
			    void* input1, void* input2, void* output);

			template <typename T>
			void add_kernel_implementation_avx(
			    void* input1, void* input2, void* output);

			template <typename T>
			void add_kernel_implementation_avx2(
			    void* input1, void* input2, void* output);

			template <typename T>
			void add_kernel_implementation_sse2(
			    void* input1, void* input2, void* output);

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
				std::tuple<BinaryPacketFunction, RegisterType> kernel_implementation_packet() const
				{
					auto* CPU_arch = CPUArch::instance();
					if (CPU_arch->has_avx512f())
					{
						auto k = [](const Packet& packet1, const Packet& packet2, const Packet& output_packet){
							add_kernel_implementation_avx512f<T>((void*)&packet1, (void*)&packet2, (void*)&output_packet);
						};
						return std::make_tuple(k, get_register_type_from_instructions(CPUArch::SIMD::AVX512F));
					}
					else if (CPU_arch->has_avx2())
					{
						auto k =  [](const Packet& packet1, const Packet& packet2, const Packet& output_packet){
							add_kernel_implementation_avx2<T>((void*)&packet1, (void*)&packet2, (void*)&output_packet);
						};
						return std::make_tuple(k, get_register_type_from_instructions(CPUArch::SIMD::AVX2));
					}
					else if (CPU_arch->has_avx())
					{
						auto k = [](const Packet& packet1, const Packet& packet2, const Packet& output_packet){
							add_kernel_implementation_avx<T>((void*)&packet1, (void*)&packet2, (void*)&output_packet);
						};
						return std::make_tuple(k, get_register_type_from_instructions(CPUArch::SIMD::AVX));
					}
					else if (CPU_arch->has_sse2())
					{
						auto k = [](const Packet& packet1, const Packet& packet2, const Packet& output_packet){
							add_kernel_implementation_sse2<T>((void*)&packet1, (void*)&packet2, (void*)&output_packet);
						};
						return std::make_tuple(k, get_register_type_from_instructions(CPUArch::SIMD::SSE2));
					}
					else
					{
						// still have to implement GenericPackets, i.e. represent a scalar
						auto k = [](const Packet& packet1, const Packet& packet2, const Packet& output_packet){
							// std::transform(
							//     std::get<T*>(packet1.m_data),
							//     std::get<T*>(packet1.m_data) + size,
							//     std::get<T*>(packet2), std::get<T*>(output_packet),
							//     std::plus<T>());
						};
						return std::make_tuple(k, get_register_type_from_instructions(CPUArch::SIMD::NONE));
					}
				}

				template <typename T>
				void kernel_implementation(
				    void* input1, void* input2, void* output, const size_t size)
				{
					const auto [kernel, register_type] = kernel_implementation_packet<T>();
					if (register_type != RegisterType::SCALAR)
					{
						const size_t register_capacity = static_cast<size_t>(register_type) / sizeof(T);
						size_t i = 0;
						if (size > register_capacity)
						{
							size_t remainder = size % register_capacity;
							for (;i<size-remainder; i+=register_capacity)
							{
								const auto packet1 = Packet(static_cast<const T*>(input1)+i, register_type);
								const auto packet2 = Packet(static_cast<const T*>(input2)+i, register_type);
								if constexpr(!std::is_same_v<T, bool>)
								{
									Packet output_packet = Packet(register_type);
									kernel(packet1, packet2, output_packet);
									output_packet.store(static_cast<T*>(output)+i);	
								}
								else
								{
									Packet output_packet = Packet(static_cast<T*>(output)+i, register_type);
									kernel(packet1, packet2, output_packet);
								}
							}
						}
						while(i < size)
						{
							*(static_cast<T*>(output)+i) = *(static_cast<const T*>(input1)+i) + *(static_cast<const T*>(input2)+i); 
							i++;
						}
					}
					else
					{
						std::transform(
							    static_cast<const T*>(input1),
							    static_cast<const T*>(input1) + size,
							    static_cast<const T*>(input2), static_cast<T*>(output),
							    std::plus<T>());
					}
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