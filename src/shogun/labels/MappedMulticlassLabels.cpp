/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <shogun/labels/MappedMulticlassLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/base/range.h>
#include <shogun/util/converters.h>
#include <set>

using namespace shogun;

int32_t CMappedMulticlassLabels::get_num_classes()
{
	std::set<float64_t> s;
	for (auto i : range(get_num_labels()))
		s.insert(get_label(i));
	return utils::safe_convert<int32_t>(s.size());
}

CLabels* CMappedMulticlassLabels::shallow_subset_copy()
{
	return nullptr;
}

std::pair<LabelMap, LabelMap> CMappedMulticlassLabels::create_mapping(const CLabels* orig) const
{
	LabelMap to_internal;
	LabelMap from_internal;

	switch (orig->get_label_type())
	{
		case LT_BINARY:
		case LT_MULTICLASS:
		case LT_REGRESSION:
		{
			auto dense = static_cast<const CDenseLabels*>(orig);
			auto unique = dense->get_labels().unique();
			for (auto i : range(unique.size()))
			{
				to_internal[unique[i]] = i;
				from_internal[i] = unique[i];
			}
			break;
		}
		default:
			SG_ERROR("Cannot use %s as %s.", orig->get_name(), get_name());
	}

	return std::make_pair(to_internal, from_internal);
}
