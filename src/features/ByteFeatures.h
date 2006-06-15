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

#ifndef _BYTEFEATURES__H__
#define _BYTEFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CByteFeatures: public CSimpleFeatures<BYTE>
{
	public:
		CByteFeatures(LONG size);
		CByteFeatures(const CByteFeatures & orig);
		CByteFeatures(CHAR* fname);

		virtual EFeatureType get_feature_type() { return F_BYTE; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
};
#endif
