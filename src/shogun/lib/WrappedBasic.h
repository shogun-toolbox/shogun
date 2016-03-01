/*
 * -*- coding: utf-8 -*-
 * vim: set fileencoding=utf-8
 *
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Heiko Strathmann
 */
#ifndef WRAPPED_BASIC_H__
#define WRAPPED_BASIC_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGString.h>

namespace
{
    template <typename T>
    T default_value()
    {
        return T();
    }

    template <>
    char default_value()
    {
        return '0';
    }
}

namespace shogun
{

/** @brief Simple wrapper class that allows to store any Shogun basic parameter
 * (i.e. float64_t, int64_t, char, etc) in a CSGObject, and therefore to
 * make it serializable. Using a template argument that is not a Shogun
 * parameter will cause a compile error when trying to register the passed
 * value as a parameter in the constructors.
 */
template<class T> class CWrappedBasic: public CSGObject
{
public:
	/** Default constructor. Do not use. */
	CWrappedBasic() : CSGObject()
	{
		m_value = ::default_value<T>();
		m_value_name = "Unnamed";
		set_generic<T>();
		register_params();
	}

	/** Constructor.
	 * @param value Value to wrap as CSGObject.
	 * @param value_name Name under which value is registered.
	*/
	CWrappedBasic(T value, const char* value_name="")
	{
		m_value = value;
		m_value_name = value_name;
		set_generic<T>();
		register_params();
	}

	/** @return name of the CSGObject, without C prefix */
	virtual const char* get_name() const { return "WrappedBasic"; }

private:
	void register_params()
	{
		SG_ADD(&m_value, m_value_name, "Wrapped value", MS_NOT_AVAILABLE);
	}

protected:
	/** Wrapped value. */
	T m_value;

	/** Name of wrapped value */
	const char* m_value_name;
};

template class CWrappedBasic<bool>;
template class CWrappedBasic<char>;
template class CWrappedBasic<int8_t>;
template class CWrappedBasic<uint8_t>;
template class CWrappedBasic<int16_t>;
template class CWrappedBasic<uint16_t>;
template class CWrappedBasic<int32_t>;
template class CWrappedBasic<uint32_t>;
template class CWrappedBasic<int64_t>;
template class CWrappedBasic<uint64_t>;
template class CWrappedBasic<float32_t>;
template class CWrappedBasic<float64_t>;
template class CWrappedBasic<floatmax_t>;

};
#endif // WRAPPED_BASIC_H__
