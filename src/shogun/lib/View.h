/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Wuwei Lin
 */
#ifndef _VIEW__H__
#define _VIEW__H__

#include <shogun/features/Features.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/SGVector.h>
#include <type_traits>

namespace shogun
{

	template <class T>
	T* view(T* viewable, const SGVector<index_t>& subset)
	{
		static_assert(
		    std::is_base_of<CFeatures, T>::value ||
		        std::is_base_of<CLabels, T>::value,
		    "Only CFeatures and CLabels are viewable.");
		auto result = viewable->duplicate();
		result->add_subset(subset);
		return static_cast<T*>(result);
	}

	template <class T>
	T* view(Some<T> viewable, const SGVector<index_t>& subset)
	{
		return view(viewable.get(), subset);
	}
} // namespace shogun

#endif
