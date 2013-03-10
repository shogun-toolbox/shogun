/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_METHOD_TRAITS_H_
#define TAPKEE_METHOD_TRAITS_H_

namespace tapkee
{

//! Traits used to obtain information about dimension reduction methods compile-time
//!
//! Usage: 
//! \code
//! MethodTraits<SomeDimensionReductionMethod>::some_information() 
//! \endcode
template <int method> struct MethodTraits
{
	//! @return true if method needs kernel callback
	static bool needs_kernel();
	//! @return true if method needs distance callback
	static bool needs_distance();
	//! @return true if method needs feature vector access callback
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

#endif
