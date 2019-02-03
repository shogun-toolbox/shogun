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

#ifndef STICHWORT_EXCEPTIONS_H_
#define STICHWORT_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace stichwort
{

//! An exception type that is thrown in case of missed parameter,
//! i.e. when some required parameter is not set.
class missed_parameter_error : public std::logic_error
{
	public:
		/** @param what_msg message of the exception */
		explicit missed_parameter_error(const std::string& what_msg) :
			std::logic_error(what_msg) {};
};

//! An exception type that is thrown in case if wrong parameter
//! value is passed.
class wrong_parameter_error : public std::logic_error
{
	public:
		/** @param what_msg message of the exception */
		explicit wrong_parameter_error(const std::string& what_msg) :
			std::logic_error(what_msg) {};
};

//! An exception type that is thrown in case if wrong parameter
//! value is passed.
class wrong_parameter_type_error : public std::logic_error
{
	public:
		/** @param what_msg message of the exception */
		explicit wrong_parameter_type_error(const std::string& what_msg) :
			std::logic_error(what_msg) {};
};

//! An exception type that is thrown when some parameter is passed more than once
class multiple_parameter_error : public std::runtime_error
{
	public:
		/** @param what_msg message of the exception */
		explicit multiple_parameter_error(const std::string& what_msg) :
			std::runtime_error(what_msg) {};
};

}
#endif

