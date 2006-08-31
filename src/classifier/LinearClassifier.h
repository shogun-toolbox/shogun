/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LINEARCLASSIFIER_H__
#define _LINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "classifier/Classifier.h"

#include <stdio.h>

class CLinearClassifier : public CClassifier
{
	public:
		CLinearClassifier();
		virtual ~CLinearClassifier();

		/// get output for example "idx"
		virtual inline DREAL classify_example(INT idx)
		{
			INT vlen;
			bool vfree;
			double* vec=features->get_feature_vector(idx, vlen, vfree);
			DREAL result=CMath::dot(w,vec,vlen);
			features->free_feature_vector(vec, idx, vfree);

			return result+bias;
		}

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		virtual inline void set_features(CRealFeatures* feat) { features=feat; }
		virtual CRealFeatures* get_features() { return features; }

	protected:
		DREAL* w;
		DREAL bias;
		CRealFeatures* features;
};
#endif
