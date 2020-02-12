/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef SHOGUNSHAPE_H_
#define SHOGUNSHAPE_H_

namespace shogun
{
	class Shape
	{
	public:
		static constexpr size_t DYNAMIC = std::numeric_limits<size_t>::max();

		Shape(const std::vector<size_t>& shape) : m_shape(shape)
		{
		}
		~Shape();

		const std::vector<size_t>& get() const
		{
			return m_shape;
		}

	private:
		std::vector<size_t> m_shape;
	};
} // namespace shogun

#endif