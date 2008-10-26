/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2006 Fabio De Bona
 * Written (W) 2006 Konrad Rieck
 * Written (W) 2006-2008 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
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

/// Type SHORT is 2 bytes in size
typedef short int SHORT;

#ifndef SUNOS
#include <stdint.h>
#else
typedef long int64_t;
typedef unsigned long uint64_t;
typedef unsigned long long int uintmax_t;
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef int int32_t;
#endif


/// Type SHORTREAL is 4 bytes in size
typedef float SHORTREAL;

/// Type DREAL is 8 bytes in size
typedef double DREAL;

/// Type LONGREAL is 16 bytes in size
typedef long double LONGREAL;

#ifdef USE_SHORTREAL_KERNELCACHE
	typedef SHORTREAL KERNELCACHE_ELEM;
#else
	typedef DREAL KERNELCACHE_ELEM;
#endif

typedef int64_t KERNELCACHE_IDX;

//@}

//#define TMP_DIR "/tmp/"

#endif
