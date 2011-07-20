/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

%define DOCSTR
"The `Mathematics` module gathers all math related Objects in the SHOGUN toolkit."
%enddef

/*%module(docstring=DOCSTR) Mathematics*/
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Mathematics_doxygen.i"
#endif
#endif

%rename(Math) CMath;

%include <shogun/mathematics/Math.h>
