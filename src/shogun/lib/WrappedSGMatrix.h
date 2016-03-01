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
#ifndef WRAPPED_SGMATRIX_H__
#define WRAPPED_SGMATRIX_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGMatrix.h>


namespace shogun
{

/** @brief Simple wrapper class that allows to store any Shogun SGMatrix<T>
 * in a CSGObject, and therefore to make it serializable. Using a template
 * argument that is not a Shogun parameter will cause a compile error when
 * trying to register the passed value as a parameter in the constructors.
 */
template<class T> class CWrappedSGMatrix: public CSGObject
{
public:
	/** Default constructor. Do not use. */
	CWrappedSGMatrix() : CSGObject()
	{
		m_value_name = "Unnamed";
		set_generic<T>();
		register_params();
	}

	/** Constructor.
	 * @param value Value to wrap as CSGObject.
	 * @param value_name Name under which value is registered.
	*/
	CWrappedSGMatrix(SGMatrix<T> value, const char* value_name="")
	{
		m_value = value;
		m_value_name = value_name;
		set_generic<T>();
		register_params();
	}

	/** @return name of the CSGObject, without C prefix */
	virtual const char* get_name() const { return "WrappedSGMatrix"; }

private:
	void register_params()
	{
		SG_ADD(&m_value, m_value_name, "Wrapped value", MS_NOT_AVAILABLE);
	}

protected:
	/** Wrapped value. */
	SGMatrix<T> m_value;

	/** Name of wrapped value */
	const char* m_value_name;
};

template class CWrappedSGMatrix<bool>;
template class CWrappedSGMatrix<char>;
template class CWrappedSGMatrix<int8_t>;
template class CWrappedSGMatrix<uint8_t>;
template class CWrappedSGMatrix<int16_t>;
template class CWrappedSGMatrix<uint16_t>;
template class CWrappedSGMatrix<int32_t>;
template class CWrappedSGMatrix<uint32_t>;
template class CWrappedSGMatrix<int64_t>;
template class CWrappedSGMatrix<uint64_t>;
template class CWrappedSGMatrix<float32_t>;
template class CWrappedSGMatrix<float64_t>;
template class CWrappedSGMatrix<floatmax_t>;

};
#endif // WRAPPED_SGMATRIX_H__
