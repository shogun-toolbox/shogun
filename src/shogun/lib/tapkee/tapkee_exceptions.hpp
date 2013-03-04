/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (c) 2013 Sergey Lisitsyn
 *
 */

#ifndef TAPKEE_EXCEPTIONS_H_
#define TAPKEE_EXCEPTIONS_H_

#include <stdexcept>
#include <string>

using std::string;

namespace tapkee
{

class wrong_parameter_error : public std::logic_error
{
	public: 
		explicit wrong_parameter_error(const string& what_msg) : std::logic_error(what_msg) {};
};

class missed_parameter_error : public std::logic_error
{
	public:
		explicit missed_parameter_error(const string& what_msg) : std::logic_error(what_msg) {};
};

class unsupported_method_error : public std::logic_error
{
	public:
		explicit unsupported_method_error(const string& what_msg) : std::logic_error(what_msg) {};
};

class not_enough_memory_error : public std::runtime_error
{
	public:
		explicit not_enough_memory_error(const string& what_msg) : std::runtime_error(what_msg) {};
};

class eigendecomposition_error : public std::runtime_error
{
	public:
		explicit eigendecomposition_error(const string& what_msg) : std::runtime_error(what_msg) {};
};

}
#endif

