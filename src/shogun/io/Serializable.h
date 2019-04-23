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
#include <shogun/lib/SGStringList.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif

/** @brief A trait that makes a none SGObject SG-serializable
 * This only works with classes derived of SGReferencedData (SGVector, SGMatrix,
 * SGStringList, etc) and fundamental types.
 */
template<class T> class Serializable: public SGObject
{
public:
	/** Default constructor. Do not use. */
	Serializable() : SGObject()
	{
		init();
	}

	/** Constructor.
	 * @param value Value to serialize as SGObject.
	 * @param value_name Name under which value is registered.
	*/
	Serializable(T value, const char* value_name=""): SGObject(), m_name(value_name)
	{
		init();
		m_value = value;
	}

	/**
	 * Get stored value
	 *
	 * @return stored value
	 */
	virtual T get_value() const { return m_value; }

	/** @return name of the SGObject, without C prefix */
	virtual const char* get_name() const { return "Serializable"; }

private:
	void init()
	{
		set_generic<typename extract_value_type<T>::value_type>();
		m_value = 0;
		SG_ADD(&m_value, "value", "Serialized value");
		watch_param("name", &m_name);
	}

protected:
	/** Serialized value. */
	T m_value;
	std::string m_name;
};

// FIXME: once the class factory is refactored this should be dropped and
// Serializable should be use directly
template<class T> class VectorSerializable: public Serializable<SGVector<T>>
{
public:
	VectorSerializable() : Serializable<SGVector<T>>() {}
	VectorSerializable(SGVector<T> value, const char* value_name=""): Serializable<SGVector<T>>(value, value_name) {}
	virtual ~VectorSerializable() {}

	/** @return name of the SGObject, without C prefix */
	virtual const char* get_name() const { return "VectorSerializable"; }
};

template<class T> class MatrixSerializable: public Serializable<SGMatrix<T>>
{
public:
	MatrixSerializable() : Serializable<SGMatrix<T>>() {}
	MatrixSerializable(SGMatrix<T> value, const char* value_name=""): Serializable<SGMatrix<T>>(value, value_name) {}
	virtual ~MatrixSerializable() {}

	/** @return name of the SGObject, without C prefix */
	virtual const char* get_name() const { return "MatrixSerializable"; }
};

// FIXME: there is no SG_ADD for SGStringList so need to do that manually.
// can be dropped once SG_ADD works with SGStringList
// Note: cannot inherit from Serializable as need to overload/change init()
template<class T> class StringListSerializable: public SGObject
{
public:
	StringListSerializable() : SGObject() { init(); }
	StringListSerializable(SGStringList<T> value, const char* value_name="") : SGObject()
	{
		init();
		m_value = value;
	}
	virtual ~StringListSerializable() {}
	virtual const char* get_name() const { return "StringListSerializable"; }

protected:
	virtual void init()
	{
		set_generic<T>();
		m_value = SGStringList<T>();
		m_parameters->add_vector(&m_value.strings, &m_value.num_strings, "value",
				"Serialized value");
	}

	SGStringList<T> m_value;
};
};
#endif // SHOGUN_SERIALIZABLE_H_
