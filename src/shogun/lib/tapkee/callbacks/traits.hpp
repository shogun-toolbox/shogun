/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_TRAITS_H_
#define TAPKEE_TRAITS_H_

/** Callback traits used to indicate
 * whether provided pairwise callback is
 * a kernel (similarity function) or a
 * distance function
 */
template <class Callback>
struct BasicCallbackTraits
{
	bool is_kernel();
	bool is_linear_kernel();
	bool is_distance();
};

/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_CALLBACK_IS_KERNEL(X) template<> struct BasicCallbackTraits<X> \
{ \
	static bool is_kernel() { return true; }; \
	static bool is_linear_kernel() { return false; }; \
	static bool is_distance() { return false; }; \
};
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_KERNEL(X) TAPKEE_CALLBACK_IS_KERNEL(X)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_KERNEL_CALLBACK(X) TAPKEE_CALLBACK_IS_KERNEL(X)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X) template<> struct BasicCallbackTraits<X> \
{ \
	static bool is_kernel() { return true; }; \
	static bool is_linear_kernel() { return true; }; \
	static bool is_distance() { return true; }; \
};
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_LINEAR_KERNEL(X) TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X)
/** Macro used to indicate that callback X is a kernel function */
#define TAPKEE_IS_LINEAR_KERNEL_CALLBACK(X) TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X)
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_CALLBACK_IS_DISTANCE(X) template<> struct BasicCallbackTraits<X> \
{ \
	static bool is_kernel() { return false; }; \
	static bool is_linear_kernel() { return false; }; \
	static bool is_distance() { return true; }; \
};
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_IS_DISTANCE(X) TAPKEE_CALLBACK_IS_DISTANCE(X)
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_IS_DISTANCE_CALLBACK(X) TAPKEE_CALLBACK_IS_DISTANCE(X)

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
