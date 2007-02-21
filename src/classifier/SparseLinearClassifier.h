/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSELINEARCLASSIFIER_H__
#define _SPARSELINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/SparseFeatures.h"
#include "classifier/Classifier.h"

class CSparseLinearClassifier : public CClassifier
{
	public:
		CSparseLinearClassifier();
		virtual ~CSparseLinearClassifier();

		/// get output for example "idx"
		virtual inline DREAL classify_example(INT idx)
		{
			//INT vlen;
			//bool vfree;
			//double* vec=features->get_feature_vector(idx, vlen, vfree);
			//DREAL result=CMath::dot(w,vec,vlen);
			//features->free_feature_vector(vec, idx, vfree);

			//return result+bias;
			return idx;
		}

        //inline void get_w(DREAL** dst_w, INT* dst_dims)
        //{
        //    ASSERT(dst_w && dst_dims);
        //    ASSERT(w && features);
        //    *dst_dims=features->get_num_features();
        //    *dst_w=new DREAL[*dst_dims];
        //    ASSERT(*dst_w);
        //    memcpy(*dst_w, w, sizeof(DREAL) * (*dst_dims));
        //}

        inline DREAL get_bias()
        {
            return bias;
        }

		inline void set_features(CSparseFeatures<DREAL>* feat) { features=feat; }
		inline CSparseFeatures<DREAL>* get_features() { return features; }

	protected:
		DREAL* w;
		DREAL bias;
		CSparseFeatures<DREAL>* features;
};
#endif
