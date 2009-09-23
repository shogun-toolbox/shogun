/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
#ifdef HAVE_BOOST_SERIALIZATION

#include "features/StringFeatures.h"
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT(CStringFeatures<char>);

#endif //HAVE_BOOST_SERIALIZATION

