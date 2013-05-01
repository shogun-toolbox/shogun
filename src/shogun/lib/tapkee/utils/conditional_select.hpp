/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn, Fernando Iglesias
 */

#ifndef TAPKEE_CONDITIONAL_SELECT_H_
#define TAPKEE_CONDITIONAL_SELECT_H_

namespace tapkee
{
namespace tapkee_internal
{

template<bool, typename T>
struct conditional_select
{
	inline T operator()(T a, T b) const;
};

template<typename T>
struct conditional_select<true,T>
{
	inline T operator()(T a, T) const
	{
		return a;
	}
};

template<typename T>
struct conditional_select<false,T>
{
	inline T operator()(T, T b) const
	{
		return b;
	}
};

}
}

#endif
