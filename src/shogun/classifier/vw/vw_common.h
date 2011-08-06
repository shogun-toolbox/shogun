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

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

#include <shogun/lib/v_array.h>
#include <shogun/classifier/vw/substring.h>
#include <shogun/classifier/vw/vw_environment.h>
#include <shogun/classifier/vw/vw_label.h>
#include <shogun/classifier/vw/vw_example.h>

namespace shogun
{

using std::string;

typedef size_t (*hash_func_t)(substring, unsigned long);

const int quadratic_constant = 27942141;
const int constant = 11650396;

}
#endif // _VW_COMMON_H__
