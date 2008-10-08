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

/// Type CHAR
typedef char CHAR;
typedef CHAR* P_CHAR;

/// Type BYTE 
typedef unsigned char BYTE;
typedef BYTE* P_BYTE;

/// Type SHORT is 2 bytes in size
typedef short int SHORT;
typedef SHORT* P_SHORT;

/// Type WORD is 2 bytes in size
typedef unsigned short int WORD;
typedef WORD* P_WORD;

/// Type INT is 4 bytes in size
typedef int INT;
typedef INT* P_INT;

/// Type INT is 4 bytes in size
typedef unsigned int UINT;
typedef UINT* P_UINT;

/// Type LONG is 8 bytes in size
#ifndef SUNOS
#include <stdint.h>
typedef int64_t LONG;
#else
typedef long LONG;
#endif
typedef LONG* P_LONG;

/// Type ULONG is 8 bytes in size
#ifndef SUNOS
#include <stdint.h>
typedef uint64_t ULONG;
#else
typedef unsigned long ULONG;
#endif
typedef ULONG* P_ULONG;

/// Type SHORTREAL is 4 bytes in size
typedef float SHORTREAL;
typedef SHORTREAL* P_SHORTREAL;

/// Type DREAL is 8 bytes in size
typedef double DREAL;
typedef DREAL* P_DREAL;

/// Type LONGREAL is 16 bytes in size
typedef long double LONGREAL;
typedef LONGREAL* P_LONGREAL;

#ifdef USE_SHORTREAL_KERNELCACHE
	typedef SHORTREAL KERNELCACHE_ELEM;
#else
	typedef DREAL KERNELCACHE_ELEM;
#endif

typedef LONG KERNELCACHE_IDX;

//@}

//#define TMP_DIR "/tmp/"

#endif
