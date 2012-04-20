/*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* Written (W) 2012 Philippe Tillet
*/

#ifdef USE_OPENCL

#ifndef VIENNACL_COMPATIBILITY_HPP
#define VIENNACL_COMPATIBILITY_HPP

#include <shogun/features/FeatureTypes.h>

namespace shogun{

		template<class ST>
		struct make_vcl_compatible{
			typedef float Result;
		};

		template<>
		struct make_vcl_compatible<double>{
			typedef double Result;
		};
}

#endif

#endif