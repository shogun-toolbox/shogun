/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNTENSOR_H_
#define SHOGUNTENSOR_H_

#include <shogun/mathematics/graph/Allocator.h>
#include <shogun/mathematics/graph/Types.h>

#include <numeric>

namespace shogun
{
	// potential entry point to unify backend memory representations?
	// might have to use void*& and store datatype
	class Tensor
	{
	public:
		template <typename T>
		Tensor(const SGVector<T>& vec) : m_data(vec.vec), m_shape({vec.size()})
		{
		}

		static Tensor
		create_empty(const std::vector<size_t>& shape, element_type type)
		{
			auto size = std::accumulate(
			    shape.begin(), shape.end(), 1, std::multiplies{});
			void* data = allocator_dispatch(size, type);
			return Tensor(data, shape);
		}

		std::vector<size_t> get_shape() const
		{
			return m_shape;
		}

		size_t get_size() const
		{
			return std::accumulate(
			    m_shape.begin(), m_shape.end(), 1, std::multiplies{});
		}

		void*& data()
		{
			return m_data;
		}

	protected:
		Tensor(void* data, const std::vector<size_t>& shape)
		    : m_shape(shape), m_data(data)
		{
		}

	private:
		void* m_data;
		const std::vector<size_t> m_shape;
	};
} // namespace shogun

#endif