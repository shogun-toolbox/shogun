/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_COMMON_H__
#define _VW_COMMON_H__

#include <shogun/lib/DataType.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#include <shogun/lib/v_array.h>
#include <shogun/io/SGIO.h>
#include <shogun/classifier/vw/VwEnvironment.h>
#include <shogun/classifier/vw/vw_label.h>
#include <shogun/classifier/vw/vw_example.h>

namespace shogun
{
/// Hash function typedef, takes a substring and seed as parameters
typedef uint32_t (*hash_func_t)(substring, uint32_t);

/// Constant used while hashing/accessing quadratic features
const int32_t quadratic_constant = 27942141;

/// Constant used to access the constant feature
const int32_t constant_hash = 11650396;

/// Seed for hash
const uint32_t hash_base = 97562527;

}
#endif // _VW_COMMON_H__
