/** Stichwort
 *
 * Copyright (c) 2013, Sergey Lisitsyn <lisitsyn.s.o@gmail.com>
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef STICHWORT_KEYWORDS_H_
#define STICHWORT_KEYWORDS_H_

#include <shogun/lib/tapkee/stichwort/parameter.hpp>

//! The namespace that contains implementations for the keywords
namespace stichwort
{
	/** DefaultValue instance is useful
	 * to set a parameter its default value.
	 *
	 * Once assigned to a keyword it produces a parameter
	 * with the default value assigned to the keyword.
	 */
	struct DefaultValue
	{
		DefaultValue() { }
	};

	/** ParameterKeyword instance is used to represent
	 * a keyword that is assigned to some value. Such
	 * an assignment results to instance of @ref Parameter
	 * class which can be later checked and casted back
	 * to the value it represents.
	 *
	 * Usage is
	 * @code
	 * 	ParameterKeyword<int> keyword;
	 * 	Parameter p = (keyword = 5);
	 * 	int p_value = p;
	 * @endcode
	 */
	template <typename T>
	struct ParameterKeyword
	{
		typedef std::string Name;
		typedef T Type;

		ParameterKeyword(const Name& n, const T& dv) : name(n), default_value(dv) { }
		ParameterKeyword(const ParameterKeyword& pk);
		ParameterKeyword operator=(const ParameterKeyword& pk);

		Parameter operator=(const T& value) const
		{
			return Parameter::create(name,value);
		}
		Parameter operator=(const DefaultValue&) const
		{
			return Parameter::create(name,default_value);
		}
		operator Name() const
		{
			return name;
		}

		Name name;
		T default_value;
	};

	struct ParametersForwarder
	{
		ParametersForwarder()
		{

		}
		ParametersForwarder(const ParametersForwarder&);
		ParametersForwarder& operator=(const ParametersForwarder&);
		ParametersSet operator[](ParametersSet parameters) const
		{
			return parameters;
		}
	};

	namespace
	{
		/** The default value - assigning any keyword to this
		 * static struct produces a parameter with its default value.
		 */
		const DefaultValue by_default;

		const ParametersForwarder kwargs;
	}
}
#endif
