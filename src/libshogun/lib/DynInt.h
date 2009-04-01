/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __DYNINT_H__
#define __DYNINT_H__

#include <shogun/lib/common.h>
#include <shogun/lib/io.h>

/** @brief integer type of dynamic size
 *
 * This object can be used to create huge integers. These integers can be used
 * directly instead of the usual int32_t etc types since operators are properly
 * overloaded.
 *
 * An exampe use would be 512 wide unsigned ints consisting of four uint64's:
 *
 * CDynInt<uint64_t, 4> int512;
 *
 * This data type is mostly used as a (efficient) storage container for
 * bit-mapped strings. Therefore, currently only comparison, assignment and
 * bit operations are implemented.
 *
 * \todo implement add,mul,div
 */
template <class T, int sz> class CDynInt
{
public:
	/** default constructor
	 *
	 * creates a DynInt that is all zero.
	 */
	CDynInt()
	{
		for (int i=0; i<sz; i++)
			integer[i]=0;
	}

	/** constructor (set least significant ``word'')
	 *
	 * The least significant word is set, the rest filled with zeros.
	 *
	 * @param x least significant word
	 */
	CDynInt(T x)
	{
		for (int i=0; i<sz-1; i++)
			integer[i]=0;
		integer[sz-1]=x;
	}

	/** constructor (set whole array)
	 *
	 * Initialize the DynInt based on an array, which is passed as an argument.
	 *
	 * @param x array of size sz
	 */
	CDynInt(const T x[sz])
	{
		for (int i=0; i<sz; i++)
			integer[i]=x[i];
	}

	/** copy constructor */
	CDynInt(const CDynInt<T,sz> &x)
	{
		for (int i=0; i<sz; i++)
			integer[i]=x.integer[i];
	}

	/** destructor */
	~CDynInt()
	{
	}

	/** overload = operator
	 * @param x assign elements from x
	 */
	CDynInt<T,sz>& operator=(const CDynInt<T,sz>& x)
	{ 
		for (int i=0; i<sz; i++)
			integer[i]=x.integer[i];
		return *this; 
	}

	/** overload | operator and return x | y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator|(const CDynInt<T,sz>& x) const
	{
		CDynInt<T,sz> r;

		for (int i=sz-1; i>=0; i--)
			r.integer[i]=integer[i] | x.integer[i];

		return r;
	}

	/** overload & operator and return x & y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator&(const CDynInt<T,sz>& x) const
	{
		CDynInt<T,sz> r;

		for (int i=sz-1; i>=0; i--)
			r.integer[i]=integer[i] & x.integer[i];

		return r;
	}

	/** overload ^ operator and return x ^ y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator^(const CDynInt<T,sz>& x) const
	{
		CDynInt<T,sz> r;

		for (int i=sz-1; i>=0; i--)
			r.integer[i]=integer[i] ^ x.integer[i];

		return r;
	}

	/** overload + operator and return x + y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator+(const CDynInt<T,sz> &x) const
	{
		CDynInt<T,sz> r;

		T overflow=0;
		for (int i=sz-1; i>=0; i--)
		{
			r.integer[i]=integer[i]+x.integer[i]+overflow;
			if (r.integer[i] < CMath::max(integer[i], x.integer[i]))
				overflow=1;
			else
				overflow=0;
		}

		return x;
	}

	/** overload - operator and return x - y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator-(const CDynInt<T,sz> &x) const
	{
		return NULL;
	}

	/** overload / operator and return x / y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator/(const CDynInt<T,sz> &x) const
	{
		return NULL;
	}

	/** overload * operator and return x * y 
	 *
	 * @param x x
	 */
	const CDynInt<T,sz> operator*(const CDynInt<T,sz> &x) const
	{
		return NULL;
	}

	/** overload += operator; add x to current DynInt
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator+=(const CDynInt<T,sz> &x)
	{
		return NULL;
	}

	/** overload -= operator; substract x from current DynInt
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator-=(const CDynInt<T,sz> &x)
	{
		return NULL;
	}

	/** overload *= operator; multiple x to with current DynInt
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator*=(const CDynInt<T,sz> &x)
	{
		return NULL;
	}

	/** overload /= operator; divide current object by x
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator/=(const CDynInt<T,sz> &x)
	{
		return NULL;
	}

	/** overload == operator; test if current object equals x
	 *
	 * @param x x
	 */
	bool operator==(const CDynInt<T,sz> &x) const
	{
		for (int i=sz-1; i>=0; i--)
		{
			if (integer[i]!=x.integer[i])
				return false;
		}

		return true;
	}

	/** overload >= operator; test if current object greater equal x
	 *
	 * @param x x
	 */
	bool operator>=(const CDynInt<T,sz> &x) const
	{
		for (int i=0; i<sz; i++)
		{
			if (integer[i]>x.integer[i])
				return true;
			if (integer[i]<x.integer[i])
				return false;
		}
		return true;
	}

	/** overload <= operator; test if current object lower equal x
	 *
	 * @param x x
	 */
	bool operator<=(const CDynInt<T,sz> &x) const
	{
		for (int i=0; i<sz; i++)
		{
			if (integer[i]<x.integer[i])
				return true;
			if (integer[i]>x.integer[i])
				return false;
		}
		return true;
	}

	/** overload > operator; test if current object is bigger than x
	 *
	 * @param x x
	 */
	bool operator>(const CDynInt<T,sz> &x) const
	{
		for (int i=0; i<sz; i++)
		{
			if (integer[i]>x.integer[i])
				return true;
			if (integer[i]<x.integer[i])
				return false;
		}
		return false;
	}

	/** overload < operator; test if current object is smaller than x
	 *
	 * @param x x
	 */
	bool operator<(const CDynInt<T,sz> &x) const
	{
		for (int i=0; i<sz; i++)
		{
			if (integer[i]<x.integer[i])
				return true;
			if (integer[i]>x.integer[i])
				return false;
		}
		return false;
	}

	/** overload ! operator; test if current object is not equal to x
	 *
	 * @param x x
	 */
	bool operator!=(const CDynInt<T,sz> &x) const
	{
		for (int i=sz-1; i>=0; i--)
		{
			if (integer[i]!=x.integer[i])
				return true;
		}
		return false;
	}

	/** overload |= operator
	 *
	 * perform bitwise or with current DynInt and x
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator|=(const CDynInt<T,sz>& x)
	{
		for (int i=sz-1; i>=0; i--)
			integer[i]|=x.integer[i];

		return *this;
	}

	/** overload &= operator
	 *
	 * perform bitwise and with current DynInt and x
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator&=(const CDynInt<T,sz>& x)
	{
		for (int i=sz-1; i>=0; i--)
			integer[i]&=x.integer[i];

		return *this;
	}

	/** overload ^= operator
	 *
	 * perform bitwise xor with current DynInt and x
	 *
	 * @param x x
	 */
	CDynInt<T,sz>& operator^=(const CDynInt<T,sz>& x)
	{
		for (int i=sz-1; i>=0; i--)
			integer[i]^=x.integer[i];

		return *this;
	}

	/** overload <<= operator
	 *
	 * perform bit shift to the left
	 *
	 * @param shift shift by this amount
	 */
	CDynInt<T,sz>& operator<<=(int shift)
	{
		T overflow=0;
		for (int i=sz-1; i>=0; i--)
		{
			integer[i]= (integer[i] << shift) | overflow;
			overflow = integer[i] >> (sizeof(T)*8 - shift);
		}

		return *this;
	}

	/** overload >>= operator
	 *
	 * perform bit shift to the right
	 *
	 * @param shift shift by this amount
	 */
	CDynInt<T,sz>& operator>>=(int shift)
	{
		T overflow=0;
		for (int i=sz-1; i>=0; i--)
		{
			integer[i]= (integer[i] >> shift) | overflow;
			overflow = (integer[i] >> (sizeof(T)*8 - shift)) << (sizeof(T)*8 - shift);
		}

		return *this;
	}

	/** negate DynInt */
	CDynInt<T,sz>& operator~()
	{
		for (int i=sz-1; i>=0; i--)
			integer[i]= ~integer[i];
		return *this;
	}

	/** cast to least significant word *dangerous* */
	operator T() { return integer[sz-1]; }

	/** decrement DynInt by one */
	CDynInt<T,sz>& operator--()
	{
		T overflow=0;
		for (int i=sz-1; i>=0; i--)
		{
			T x = integer[i]-1-overflow;
			overflow=0;
			if (integer[i]>x)
				overflow=1;
			integer[i]=x;
		}
		return *this;
	}

	/** increment DynInt by one */
	CDynInt<T,sz>& operator++()
	{
		T overflow=0;
		for (int i=sz-1; i>=0; i--)
		{
			T x = integer[i]+1+overflow;
			overflow=0;
			if (integer[i]>x)
				overflow=1;
			integer[i]=x;
		}
		return *this;
	}

	/** print the current long integer in hex (without carriage return */
	void print_hex() const
	{
		for (int i=0; i<sz; i++)
			SG_SPRINT("%.16llx", (uint64_t) integer[i]);
	}

	/** print the current long integer in bits (without carriage return */
	void print_bits() const
	{
		for (int i=0; i<sz; i++)
		{
			T mask = ((T) 1)<<(sizeof(T)*8-1);
			while (mask)
			{
				if (mask & integer[i])
					SG_SPRINT("1");
				else
					SG_SPRINT("0");

				mask>>=1;
			}
		}
	}

private:
	/** the integer requiring sizeof(T)*sz bytes */
	T integer[sz];
};

/** convenience typedefs */
typedef CDynInt<uint64_t,3> uint192_t;
typedef CDynInt<uint64_t,3> uint256_t;
typedef CDynInt<uint64_t,3> uint512_t;
typedef CDynInt<uint64_t,3> uint1024_t;
#endif // __DYNINT_H__
