/*
 * -*- coding: utf-8 -*-
 * vim: set fileencoding=utf-8
 *
 * Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
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
 */
#ifndef SHOGUN_SERIALIZABLE_H__
#define SHOGUN_SERIALIZABLE_H__

#include <shogun/base/SGObject.h>

namespace shogun
{

template<typename T>
struct extract_value_type
{
	typedef T value_type;
};

template<template<typename, typename ...> class X, typename T, typename ...Args>
struct extract_value_type<X<T, Args...>>
{
	typedef T value_type;
};

/** @brief A trait that makes a none SGObject SG-serializable
 * This only works with classes derived of SGReferencedData (SGVector, SGMatrix etc)
 * and fundamental types (std::is_arithmetic)
 */
template<class T> class CSerializable: public CSGObject
{
public:
	/** Default constructor. Do not use. */
	CSerializable() : CSGObject()
	{
		init();
	}

	/** Constructor.
	 * @param value Value to serialize as CSGObject.
	 * @param value_name Name under which value is registered.
	*/
	CSerializable(T value, const char* value_name="")
	{
		init();
		m_value = value;
		m_value_name = value_name;
	}

	/** @return name of the CSGObject, without C prefix */
	virtual const char* get_name() const { return "Serializable"; }

private:
	void init()
	{
		set_generic<typename extract_value_type<T>::value_type>();
		m_value = 0;
		m_value_name = "Unnamed";

		SG_ADD(&m_value, "value", "Serialized value", MS_NOT_AVAILABLE);
	}

protected:
	/** Serialized value. */
	T m_value;

	/** Name of serialized value */
	const char* m_value_name;
};

};
#endif // SHOGUN_SERIALIZABLE_H_
