/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSELINEARCLASSIFIER_H__
#define _SPARSELINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/SparseFeatures.h"
#include "classifier/Classifier.h"

/** class SparseLinearClassifier */
class CSparseLinearClassifier : public CClassifier
{
	public:
		/** default constructor */
		CSparseLinearClassifier();
		virtual ~CSparseLinearClassifier();

		/** classify all examples
		 *
		 * @param output resulting labels
		 * @return resulting labels
		 */
		virtual CLabels* classify(CLabels* output=NULL);

		/// get output for example "idx"
		virtual inline DREAL classify_example(INT idx)
		{
			return features->dense_dot(1.0, idx, w, w_dim, bias);
		}

		/** get w
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		inline void get_w(DREAL** dst_w, INT* dst_dims)
		{
			ASSERT(dst_w && dst_dims);
			ASSERT(w && w_dim>0);
			*dst_dims=w_dim;
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

		/** set features
		 *
		 * @param feat features to set
		 */
		inline void set_features(CSparseFeatures<DREAL>* feat) { features=feat; }

		/** get features
		 *
		 * @return features
		 */
		inline CSparseFeatures<DREAL>* get_features() { return features; }

	protected:
		/** dimension of w */
		INT w_dim;
		/** w */
		DREAL* w;
		/** bias */
		DREAL bias;
		/** features */
		CSparseFeatures<DREAL>* features;
};
#endif
