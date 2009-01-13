/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2006 Fabio De Bona
 * Written (W) 2006 Konrad Rieck
 * Written (W) 2006-2008 Christian Gehl
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdlib.h>
#include <stdio.h>
#include "lib/config.h"
#include "lib/memory.h"

#ifdef SUNOS
#define bool int
#define false 0
#define true 1
#endif

#ifndef LINUX
#define RANDOM_MAX 2147483647
#else
#define RANDOM_MAX RAND_MAX
#endif

/**@name Standard Types 
 * Definition of Platform independent Types
*/
//@{

#ifdef SUNOS
typedef char int8_t;
typedef short int int16_t;
typedef int int32_t;
typedef long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
typedef unsigned long long int uintmax_t;
#else
#include <stdint.h>
#endif


/**
 * Implementations tend to follow IEEE754
 * @see http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4610935
 */
typedef float float32_t;
typedef double float64_t;
typedef long double float128_t;

//@}

#endif
