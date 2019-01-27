/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <gtest/gtest.h>

#include <shogun/features/FeatureTypes.h>
#include <shogun/lib/sg_types.h>

using namespace shogun;
using testing::StaticAssertTypeEq;

TEST(SGTypes, get_type_from_index)
{
    StaticAssertTypeEq<getTypeFromIndex<sg_feature_types, F_UNKNOWN>::type , Unknown>();
    StaticAssertTypeEq<getTypeFromIndex<sg_feature_types, F_BOOL>::type , bool>();
    StaticAssertTypeEq<getTypeFromIndex<sg_feature_types, F_ANY>::type , Any>();
}