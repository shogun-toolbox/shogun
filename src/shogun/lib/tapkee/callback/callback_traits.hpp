/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_CALLBACK_TRAITS_H_
#define TAPKEE_CALLBACK_TRAITS_H_

/** Callback traits used to indicate
 * whether provided pairwise callback is
 * a kernel (similarity function) or a
 * distance function
 */
template <class Callback>
struct BasicCallbackTraits
{
	static const bool is_kernel;
	static const bool is_linear_kernel;
	static const bool is_distance;
	static const bool is_euclidean_distance;
};

#define TAPKEE_CALLBACK_TRAIT_HELPER(X,KERNEL,LINEAR_KERNEL,DISTANCE,EUCLIDEAN_DISTANCE) \
template<> const bool BasicCallbackTraits<X>::is_kernel = KERNEL;                                   \
template<> const bool BasicCallbackTraits<X>::is_linear_kernel = LINEAR_KERNEL;                     \
template<> const bool BasicCallbackTraits<X>::is_distance = DISTANCE;                               \
template<> const bool BasicCallbackTraits<X>::is_euclidean_distance = EUCLIDEAN_DISTANCE            \

/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_CALLBACK_IS_KERNEL(X) TAPKEE_CALLBACK_TRAIT_HELPER(X,true,false,false,false)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_KERNEL(X) TAPKEE_CALLBACK_IS_KERNEL(X)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_KERNEL_CALLBACK(X) TAPKEE_CALLBACK_IS_KERNEL(X)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X) TAPKEE_CALLBACK_TRAIT_HELPER(X,true,true,false,false)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_LINEAR_KERNEL(X) TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_LINEAR_KERNEL_CALLBACK(X) TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X)
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_CALLBACK_IS_DISTANCE(X) TAPKEE_CALLBACK_TRAIT_HELPER(X,false,false,true,false)
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_IS_DISTANCE(X) TAPKEE_CALLBACK_IS_DISTANCE(X)
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_IS_DISTANCE_CALLBACK(X) TAPKEE_CALLBACK_IS_DISTANCE(X)
/** Macro used to indicate that callback X is a euclidean distance function */
#define TAPKEE_CALLBACK_IS_EUCLIDEAN_DISTANCE(X) TAPKEE_CALLBACK_TRAIT_HELPER(X,false,false,true,false)
/** Macro used to indicate that callback X is a euclidean distance function */
#define TAPKEE_IS_EUCLIDEAN_DISTANCE(X) TAPKEE_CALLBACK_IS_EUCLIDEAN_DISTANCE(X)
/** Macro used to indicate that callback X is a euclidean distance function */
#define TAPKEE_IS_EUCLIDEAN_DISTANCE_CALLBACK(X) TAPKEE_CALLBACK_IS_EUCLIDEAN_DISTANCE(X)

template <class Callback>
struct BatchCallbackTraits
{
	static bool supports_batch();
};
#define TAPKEE_CALLBACK_SUPPORTS_BATCH(X) template<> struct BatchCallbackTraits \
{ \
	static bool supports_batch() { return true; }; \
};


#endif
