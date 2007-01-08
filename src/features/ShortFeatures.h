/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SHORTFEATURES__H__
#define _SHORTFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CShortFeatures: public CSimpleFeatures<SHORT>
{
	public:
		CShortFeatures(INT size);
		CShortFeatures(const CShortFeatures & orig);

		/** load features from file
		 * fname - filename
		 */

		CShortFeatures(CHAR* fname);

		bool obtain_from_char_features(CCharFeatures* cf, INT start, INT order, INT gap=0);

		virtual EFeatureType get_feature_type() { return F_SHORT; }

		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
	protected:
		void translate_from_single_order(SHORT* obs, INT sequence_length, INT start, INT order, INT max_val, INT gap);

};
#endif
