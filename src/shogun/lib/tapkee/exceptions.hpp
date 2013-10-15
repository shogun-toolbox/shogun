/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_EXCEPTIONS_H_
#define TAPKEE_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

namespace tapkee
{

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

//! An exception type that is thrown in case of missed parameter,
//! i.e. when some required parameter is not set.
class missed_parameter_error : public std::logic_error
{
	public:
		/** @param what_msg message of the exception */
		explicit missed_parameter_error(const std::string& what_msg) :
			std::logic_error(what_msg) {};
};

//! An exception type that is thrown when unsupported method
//! is called.
class unsupported_method_error : public std::logic_error
{
	public:
		/** @param what_msg message of the exception */
		explicit unsupported_method_error(const std::string& what_msg) :
			std::logic_error(what_msg) {};
};

//! An exception type that is thrown when the library can't get
//! enough memory.
class not_enough_memory_error : public std::runtime_error
{
	public:
		/** @param what_msg message of the exception */
		explicit not_enough_memory_error(const std::string& what_msg) :
			std::runtime_error(what_msg) {};
};

//! An exception type that is thrown when some parameter is passed more than once
class multiple_parameter_error : public std::runtime_error
{
	public:
		/** @param what_msg message of the exception */
		explicit multiple_parameter_error(const std::string& what_msg) :
			std::runtime_error(what_msg) {};
};

//! An exception type that is thrown when computations were
//! cancelled.
class cancelled_exception : public std::exception
{
	public:
		explicit cancelled_exception() :
			std::exception() {};
};

//! An exception type that is thrown when eigendecomposition
//! is failed.
class eigendecomposition_error : public std::runtime_error
{
	public:
		/** @param what_msg message of the exception */
		explicit eigendecomposition_error(const std::string& what_msg) :
			std::runtime_error(what_msg) {};
};

}
#endif

