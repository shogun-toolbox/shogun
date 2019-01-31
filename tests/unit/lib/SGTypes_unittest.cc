/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include "sg_gtest_utilities.h"

using namespace shogun;
using testing::StaticAssertTypeEq;

using testing_types = Types<
    Any, int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t,
    float32_t, float64_t, floatmax_t, complex128_t, char, bool,
    shogun::Unknown>;

TEST(SGTypes, get_type_from_index)
{
	StaticAssertTypeEq<getTypeFromIndex<testing_types, 0>::type, shogun::Any>();
	StaticAssertTypeEq<getTypeFromIndex<testing_types, 1>::type, int8_t>();
	StaticAssertTypeEq<
	    getTypeFromIndex<testing_types, 14>::type, shogun::Unknown>();
}

TEST(SGTypes, pop_by_type)
{
	StaticAssertTypeEq<
	    getTypeFromIndex<
	        typename popTypesByTypes<
	            testing_types, Types<shogun::Unknown>>::type,
	        13>::type,
	    bool>();
	StaticAssertTypeEq<
	    getTypeFromIndex<
	        typename popTypesByTypes<testing_types, Types<uint64_t>>::type,
	        6>::type,
	    int64_t>();
	StaticAssertTypeEq<
	    getTypeFromIndex<
	        typename popTypesByTypes<testing_types, Types<Any>>::type, 0>::type,
	    int8_t>();
	StaticAssertTypeEq<
	    getTypeFromIndex<
	        typename popTypesByTypes<testing_types, Types<Any, int8_t>>::type,
	        0>::type,
	    int16_t>();
}