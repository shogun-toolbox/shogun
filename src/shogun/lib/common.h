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
#include <shogun/lib/config.h>

/**@name Standard Types
 * Definition of Platform independent Types
*/
//@{

#include <stdint.h>

/* No feature test:
 * ISO C99: 7.8 Format conversion of integer types	<inttypes.h>
 *
 * If not supported make sure that your development environment is
 * supporting ISO C99!
 */
#ifdef __STDC_FORMAT_MACROS
#include <inttypes.h>
#else
#define __STDC_FORMAT_MACROS 1
#include <inttypes.h>
#undef __STDC_FORMAT_MACROS
#endif

/**
 * Implementations tend to follow IEEE754
 * @see http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4610935
 */
typedef float float32_t;
typedef double float64_t;
typedef long double floatmax_t;

//@}

#define STRING_LEN                 256
#define STRING_LEN_STR             "256"
typedef char                       string_t[STRING_LEN];

typedef int                        machine_int_t;

/** index type */
typedef int32_t index_t;

/** complex type */
#include <complex>

typedef std::complex<float64_t> complex128_t;

#include <shogun/lib/memory.h>
#endif //__COMMON_H__
