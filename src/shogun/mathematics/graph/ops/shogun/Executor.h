/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNADDSHOGUN_H_
#define SHOGUNADDSHOGUN_H_

#include <shogun/mathematics/graph/CPUArch.h>
#include "Packet.h"

namespace shogun
{
	namespace graph
	{
		namespace op
		{
			class Executor
			{
			public:
				Executor();
				~Executor();

				void operator(const std::vector<std::shared_ptr<Tensor>>& inputs, std::shared_ptr<Tensor>& output)
				{
					execute_loop();
				}

			private:
				void execute_loop(void* output)
				{
					constexpr size_t register_capacity = static_cast<size_t>(register_type) / sizeof(T);
					size_t i = 0;
					if (size > register_capacity)
					{
						size_t remainder = size % register_capacity;
						for (;i<size-remainder; i+=register_capacity)
						{
							const auto packet1 = Packet(static_cast<const T*>(input1)+i, register_type);
							const auto packet2 = Packet(static_cast<const T*>(input2)+i, register_type);
							execute_inner_loop_vectorized(packet1, packet2);
						}
					}
					while(i < size)
					{
						execute_inner_loop_unrolled(i);
					}					
				}

				void execute_inner_loop_vectorized()
				{
					for (const auto& op: m_operations)
					{
						if (std::is_same_v<element_type, element_type::BOOLEAN>)
						{
							Packet output_packet = Packet(register_type);
							op(element_type, architecture_type, (void*)&packet1, (void*)&packet2, (void*)&output_packet);
							output_packet.store(static_cast<T*>(output)+i);	
						}
						else
						{
							Packet output_packet = Packet(static_cast<T*>(output)+i, register_type);
							op(element_type, architecture_type, (void*)&packet1, (void*)&packet2, (void*)&output_packet);
						}
					}
				}

				void execute_inner_loop_unrolled(size_t i)
				{
					for (const auto& op: m_operations)
					{
						*(static_cast<T*>(output)+i) = *(static_cast<const T*>(input1)+i) + *(static_cast<const T*>(input2)+i); 
					}
				}

			};

		}
	}
}