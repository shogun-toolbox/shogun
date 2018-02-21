/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni
 */

#include <shogun/lib/ShogunException.h>
#include <shogun/lib/Signal.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace shogun;

ShogunException::ShogunException(const std::string& what_arg):
	std::exception(),
	msg(what_arg)
{
}


ShogunException::ShogunException(const char* what_arg):
	std::exception(),
	msg(what_arg)
{
}

ShogunException::ShogunException(const ShogunException& orig)
{ msg = orig.msg; }

ShogunException::~ShogunException()
{
}

const char* ShogunException::what() const noexcept
{
	return msg.c_str();
}