/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef SHOGUNTENSORMAP_H_
#define SHOGUNTENSORMAP_H_

#include "Tensor.h"

#define IGNORE_IN_CLASSLIST

namespace shogun
{
	namespace graph
	{
		template<typename T>
		class TensorMap: public Tensor
		{

		public:
			friend std::shared_ptr<Tensor>
			from_device(const std::shared_ptr<Storage>& storage);

			TensorMap(T* data, const Shape& shape):
				Tensor(shape, from<T>(), Protected{})
			{
				m_storage = Storage::create_view(data, m_shape, m_type);
			}
		};
	} // namespace graph
} // namespace shogun

#endif