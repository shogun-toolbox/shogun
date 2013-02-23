/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2012, Sergey Lisitsyn, Fernando J. Iglesias García
 *
 */

namespace tapkee
{

template <int method> struct MethodTraits
{
	static bool needs_kernel();
	static bool needs_distance();
	static bool needs_feature_vectors();
};

#define METHOD_TRAIT(X,kernel_needed,distance_needed,feature_vector_needed) template <> struct MethodTraits<X> \
{ \
	static bool needs_kernel() { return kernel_needed; } \
	static bool needs_distance() { return distance_needed; } \
	static bool needs_feature_vector() { return feature_vector_needed; } \
}

#define METHOD_THAT_NEEDS_ONLY_KERNEL_IS(X) METHOD_TRAIT(X,true,false,false)
#define METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(X) METHOD_TRAIT(X,false,true,false)
#define METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(X) METHOD_TRAIT(X,true,false,true)
#define METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(X) METHOD_TRAIT(X,false,true,true)
#define METHOD_THAT_NEEDS_ONLY_FEATURES_IS(X) METHOD_TRAIT(X,false,false,true)
#define METHOD_THAT_NEEDS_NOTHING_IS(X) METHOD_TRAIT(X,false,false,false)

}
