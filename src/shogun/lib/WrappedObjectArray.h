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
#ifndef WRAPPED_OBJECT_ARRAY_H__
#define WRAPPED_OBJECT_ARRAY_H__

#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/WrappedBasic.h>
#include <shogun/lib/WrappedSGVector.h>
#include <shogun/lib/WrappedSGMatrix.h>


namespace shogun
{

/** @brief Specialization of CDynamicObjectArray that adds methods to append
 * wrapped elements to make them serializable. Objects are wrapped through the
 * classes CWrappedBasic, CWrappedSGVector, CWrappedSGMatrix.
 */
class CWrappedObjectArray : public CDynamicObjectArray
{
public:
	CWrappedObjectArray(): CDynamicObjectArray() { }

	/** Works as CDynamicObjectArray::append_element, but accepts basic Shogun types,
	 * which are wrapped through CWrappedBasicObject.
	 *
	 * @param data Data element to append, after being wrapped.
	 * @param data_name Name of wrapped data element.
	 */
	template<class T> bool append_wrapped(T data, const char* data_name="")
	{
		return this->append_element(new CWrappedBasic<T>(data, data_name));
	}

	/** Works as CDynamicObjectArray::append_element, but accepts SGVector,
	 * which are wrapped through CWrappedSGVectorObject.
	 *
	 * @param data Data element to append, after being wrapped.
	 * @param data_name Name of wrapped data element.
	 */
	template<class T> bool append_wrapped(SGVector<T> data, const char* data_name="")
	{
		return this->append_element(new CWrappedSGVector<T>(data,
				data_name));
	}

	/** Works as CDynamicObjectArray::append_element, but accepts SGMatrix,
	 * which are wrapped through CWrappedSGVectorObject.
	 *
	 * @param data Data element to append, after being wrapped.
	 * @param data_name Name of wrapped data element.
	 */
	template<class T> bool append_wrapped(SGMatrix<T> data, const char* data_name="")
	{
		return this->append_element(new CWrappedSGMatrix<T>(data,
				data_name));
	}

	/** @return name of the CSGObject, without C prefix */
	virtual const char* get_name() const { return "WrappedObjectArray"; }


};
}
#endif // WRAPPED_OBJECT_ARRAY_H__
