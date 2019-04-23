/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Giovanni De Toni
 */

#include <shogun/lib/Signal.h>
#include <shogun/lib/exception/ShogunException.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace shogun;

ShogunException::ShogunException(const std::string& what_arg)
    : std::runtime_error(what_arg)
{
}

ShogunException::ShogunException(const char* what_arg)
	: std::runtime_error(what_arg)
{
}
