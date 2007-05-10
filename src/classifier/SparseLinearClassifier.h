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
			INT vlen;
			bool vfree;
			DREAL result=0.0;

			ASSERT(features);
			ASSERT(w && features->get_num_features()==w_dim);

			TSparseEntry<DREAL>* vec=features->get_sparse_feature_vector(idx, vlen, vfree);

			for (INT i=0; i<vlen; i++)
				result+=w[vec[i].feat_index]*vec[i].entry;

			features->free_sparse_feature_vector(vec, idx, vfree);

			return result+get_bias();
		}

        inline void get_w(DREAL** dst_w, INT* dst_dims)
        {
            ASSERT(dst_w && dst_dims);
            ASSERT(w && w_dim>0);
            *dst_dims=w_dim;
            *dst_w=new DREAL[*dst_dims];
            ASSERT(*dst_w);
            memcpy(*dst_w, w, sizeof(DREAL) * (*dst_dims));
        }

		inline void set_w(DREAL* src_w, INT src_w_dim)
		{
			w=src_w;
			w_dim=src_w_dim;
		}

        inline void set_bias(DREAL b)
        {
            bias=b;
        }
        inline DREAL get_bias()
        {
            return bias;
        }

		virtual CLabels* classify(CLabels* output=NULL);

		inline void set_features(CSparseFeatures<DREAL>* feat) { features=feat; }
		inline CSparseFeatures<DREAL>* get_features() { return features; }

	protected:
		INT w_dim;
		DREAL* w;
		DREAL bias;
		CSparseFeatures<DREAL>* features;
};
#endif
