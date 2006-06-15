/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/SparseRealFeatures.h"
#include "lib/File.h"

CFeatures* CSparseRealFeatures::duplicate() const
{
	return new CSparseRealFeatures(*this);
}

bool CSparseRealFeatures::load(CHAR* fname)
{
	return false;
}

bool CSparseRealFeatures::save(CHAR* fname)
{
	return false;
}

