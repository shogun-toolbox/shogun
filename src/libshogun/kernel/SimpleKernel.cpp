/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Christian Widmer
 * Copyright (C) 2009 Max-Planck-Society
 */

#include "kernel/SimpleKernel.h"

#ifdef HAVE_BOOST_SERIALIZATION

#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleKernel<int32_t>, "CSimpleKernel_int32");
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleKernel<int64_t>, "CSimpleKernel_int64");
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleKernel<float32_t>, "CSimpleKernel_float32");
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleKernel<float64_t>, "CSimpleKernel_float64");
#endif //HAVE_BOOST_SERIALIZATION
