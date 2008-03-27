/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LINEARCLASSIFIER_H__
#define _LINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/Labels.h"
#include "features/RealFeatures.h"
#include "classifier/Classifier.h"

#include <stdio.h>

/** class LinearClassifier */
class CLinearClassifier : public CClassifier
{
	public:
		/** default constructor */
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

		/** get w
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		inline void get_w(DREAL** dst_w, INT* dst_dims)
		{
			ASSERT(dst_w && dst_dims);
			ASSERT(w && features);
			*dst_dims=features->get_num_features();
			*dst_w=new DREAL[*dst_dims];
			ASSERT(*dst_w);
			memcpy(*dst_w, w, sizeof(DREAL) * (*dst_dims));
		}

		/** set w
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w
		 */
		inline void set_w(DREAL* src_w, INT src_w_dim)
		{
			w=src_w;
			w_dim=src_w_dim;
		}

		/** set bias
		 *
		 * @param b new bias
		 */
		inline void set_bias(DREAL b)
		{
			bias=b;
		}

		/** get bias
		 *
		 * @return bias
		 */
		inline DREAL get_bias()
		{
			return bias;
		}

		/** load from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** classify all examples
		 *
		 * @param output resulting labels
		 * @return resulting labels
		 */
		virtual CLabels* classify(CLabels* output=NULL);

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual inline void set_features(CRealFeatures* feat) { SG_REF(feat); features=feat; }

		/** get features
		 *
		 * @return features
		 */
		virtual CRealFeatures* get_features() { SG_REF(features); return features; }

	protected:
		/** dimension of w */
		INT w_dim;
		/** w */
		DREAL* w;
		/** bias */
		DREAL bias;
		/** features */
		CRealFeatures* features;
};
#endif
