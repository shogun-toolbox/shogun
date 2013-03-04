/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
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
#define TAPKEE_CALLBACK_IS_LINEAR_KERNEL(X) template<> struct BasicCallbackTraits<X> \
{ \
	static bool is_kernel() { return true; }; \
	static bool is_linear_kernel() { return true; }; \
	static bool is_distance() { return true; }; \
};
/** Macro used to indicate that callback X is a distance function */
#define TAPKEE_CALLBACK_IS_DISTANCE(X) template<> struct BasicCallbackTraits<X> \
{ \
	static bool is_kernel() { return false; }; \
	static bool is_linear_kernel() { return false; }; \
	static bool is_distance() { return true; }; \
};

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
