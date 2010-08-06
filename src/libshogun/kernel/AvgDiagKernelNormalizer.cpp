/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Christian Widmer
 * Copyright (C) 2010 Max-Planck-Society
 */


#ifdef HAVE_BOOST_SERIALIZATION
#include "kernel/Kernel.h"
#include "kernel/AvgDiagKernelNormalizer.h"
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT(shogun::CAvgDiagKernelNormalizer);
#endif //HAVE_BOOST_SERIALIZATION
