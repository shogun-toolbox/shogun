/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Christian Widmer
 * Copyright (C) 2009 Max-Planck-Society
 */


#ifdef HAVE_BOOST_SERIALIZATION
#include "features/SimpleFeatures.h"
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleFeatures<int32_t>, "CSimpleFeatures_int32");
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleFeatures<int64_t>, "CSimpleFeatures_int64");
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleFeatures<float32_t>, "CSimpleFeatures_float32");
BOOST_CLASS_EXPORT_GUID(shogun::CSimpleFeatures<float64_t>, "CSimpleFeatures_float64");
#endif //HAVE_BOOST_SERIALIZATION
