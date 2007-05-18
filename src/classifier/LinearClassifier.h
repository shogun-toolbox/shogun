/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
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

        inline void get_w(DREAL** dst_w, INT* dst_dims)
        {
            ASSERT(dst_w && dst_dims);
            ASSERT(w && features);
            *dst_dims=features->get_num_features();
            *dst_w=new DREAL[*dst_dims];
            ASSERT(*dst_w);
            memcpy(*dst_w, w, sizeof(DREAL) * (*dst_dims));
        }

		inline void set_w(DREAL* src_w, INT src_w_dim)
		{
			w=src_w;
			w_dim=src_w_dim;
		}

        inline DREAL get_bias()
        {
            return bias;
        }

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		virtual CLabels* classify(CLabels* output=NULL);

		virtual inline void set_features(CRealFeatures* feat) { features=feat; }
		virtual CRealFeatures* get_features() { return features; }

	protected:
		INT w_dim;
		DREAL* w;
		DREAL bias;
		CRealFeatures* features;
};
#endif
