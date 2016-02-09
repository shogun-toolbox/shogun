/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef PARAMETER_IMPL_H_
#define PARAMETER_IMPL_H_

#include <string>
#include <functional>

namespace shogun
{

namespace linalg
{

namespace ocl
{

/**
 * @brief Struct Parameter for wrapping up parameters to custom OpenCL operation
 * strings. Supports string type, C-style string type and all basic types of
 * parameters.
 */
struct Parameter
{

	/**
	 * Constructor that initlializes the name of the parameter.
	 *
	 * @param name The name of the parameter.
	 */
	Parameter(const char* name) : m_name(name)
	{
	}

	/**
	 * Template overloaded assignment operator for basic-types that initializes
	 * an internal to_string method which is invoked when the parameter is
	 * casted to std::string.
	 *
	 * @param value The value of the parameter.
	 */
	template <typename T>
	Parameter& operator=(const T& value)
	{
		to_string=[&value]() { return std::to_string(value); };
		return *this;
	}

	/**
	 * Overloaded assignment operator for C-style string-types that initializes
	 * an internal to_string method which is invoked when the parameter is
	 * casted to std::string.
	 *
	 * @param value The value of the parameter.
	 */
	Parameter& operator=(const char* value)
	{
		to_string=[&value]() { return std::string(value); };
		return *this;
	}

	/**
	 * Overloaded assignment operator for std::string that initializes
	 * an internal to_string method which is invoked when the parameter is
	 * casted to std::string.
	 *
	 * @param value The value of the parameter.
	 */
	Parameter& operator=(std::string value)
	{
		to_string=[&value]() { return value; };
		return *this;
	}

	/**
	 * Cast opetator to std::string
	 */
	operator std::string() const
	{
		return to_string();
	}

	/** The method to_string which is initialized by the assignment operators
	 * based on the parameter value types. This method is invoked by the cast
	 * operator to std::string
	 */
	std::function<std::string()> to_string;

	/** The name of the parameter */
	std::string m_name;
};

}

}

}
#endif // PARAMETER_IMPL_H_
