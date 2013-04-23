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
template <int method> struct MethodTraits
{
	static const bool needs_kernel;
	static const bool needs_distance;
	static const bool needs_features;
};

#define METHOD_TRAIT(X,kernel_needed,distance_needed,features_needed)			\
template <> const bool MethodTraits<X>::needs_kernel = kernel_needed;			\
template <> const bool MethodTraits<X>::needs_distance = distance_needed;		\
template <> const bool MethodTraits<X>::needs_features = features_needed		\

#define METHOD_THAT_NEEDS_ONLY_KERNEL_IS(X) METHOD_TRAIT(X,true,false,false)
#define METHOD_THAT_NEEDS_ONLY_DISTANCE_IS(X) METHOD_TRAIT(X,false,true,false)
#define METHOD_THAT_NEEDS_KERNEL_AND_FEATURES_IS(X) METHOD_TRAIT(X,true,false,true)
#define METHOD_THAT_NEEDS_DISTANCE_AND_FEATURES_IS(X) METHOD_TRAIT(X,false,true,true)
#define METHOD_THAT_NEEDS_ONLY_FEATURES_IS(X) METHOD_TRAIT(X,false,false,true)
#define METHOD_THAT_NEEDS_NOTHING_IS(X) METHOD_TRAIT(X,false,false,false)
}

#endif
