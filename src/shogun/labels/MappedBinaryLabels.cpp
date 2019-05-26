/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <shogun/labels/MappedBinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>


using namespace shogun;

std::pair<LabelMap, LabelMap>  CMappedBinaryLabels::create_mapping(const CLabels* orig) const
{
	LabelMap to_internal;
	LabelMap from_internal;

	switch (orig->get_label_type())
	{
		case LT_BINARY:
			break;
		case LT_MULTICLASS:
		case LT_REGRESSION:
		{
			auto dense = static_cast<const CDenseLabels*>(orig);
			auto unique = dense->get_labels().unique();
			REQUIRE(unique.size()<=2, "Cannot use %d label values as binary labels.\n", unique.size());
			to_internal[unique[0]] = -1;
			from_internal[-1] = unique[0];

			if (unique.size()==2)
			{
				to_internal[unique[1]] = 1;
				from_internal[+1] = unique[1];
			}
			break;
		}
		default:
			SG_ERROR("Cannot use %s as %s.", orig->get_name(), orig->get_name());
	}

	return std::make_pair(to_internal, from_internal);
}
