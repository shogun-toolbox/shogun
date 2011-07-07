/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _ONLINELINEARCLASSIFIER_H__
#define _ONLINELINEARCLASSIFIER_H__

#include "lib/common.h"
#include "features/Labels.h"
#include "features/StreamingDotFeatures.h"
#include "machine/Machine.h"

#include <stdio.h>

namespace shogun
{
/** @brief Class OnlineLinearMachine is a generic interface for linear
 * machines like classifiers which work through online algorithms.
 *
 * A linear classifier computes
 *
 *  \f[
 * 		f({\bf x})= {\bf w} \cdot {\bf x} + b
 * 	\f]
 *
 * where \f${\bf w}\f$ are the weights assigned to each feature in training
 * and \f$b\f$ the bias.
 *
 * To implement a linear classifier all that is required is to define the
 * train() function that delivers \f${\bf w}\f$ above.
 *
 * Note that this framework works with linear classifiers of arbitrary feature
 * type, e.g. dense and sparse and even string based features. This is
 * implemented by using CStreamingDotFeatures that may provide a mapping function
 * \f$\Phi({\bf x})\mapsto {\cal R^D}\f$ encapsulating all the required
 * operations (like the dot product). The decision function is thus
 *
 *  \f[
 * 		f({\bf x})= {\bf w} \cdot \Phi({\bf x}) + b.
 * 	\f]
 *
 * */
class COnlineLinearMachine : public CMachine
{
	public:
		/** default constructor */
		COnlineLinearMachine();
		virtual ~COnlineLinearMachine();

		/** get w
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		inline void get_w(float64_t*& dst_w, int32_t& dst_dims)
		{
			ASSERT(w && w_dim>0);
			dst_w=w;
			dst_dims=w_dim;
		}

		/** get w (swig compatible)
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		inline void get_w(float64_t** dst_w, int32_t* dst_dims)
		{
			ASSERT(dst_w && dst_dims);
			ASSERT(w && w_dim>0);
			*dst_dims=w_dim;
			*dst_w=(float64_t*) SG_MALLOC(sizeof(float64_t)*(*dst_dims));
			ASSERT(*dst_w);
			memcpy(*dst_w, w, sizeof(float64_t) * (*dst_dims));
		}

		/** get w
		 *
		 * @return weight vector
		 */
		inline SGVector<float64_t> get_w()
		{
			return SGVector<float64_t>(w, w_dim);
		}

		/** set w
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w
		 */
		inline void set_w(float64_t* src_w, int32_t src_w_dim)
		{
			delete[] w;
			w=new float64_t[src_w_dim];
			memcpy(w, src_w, size_t(src_w_dim)*sizeof(float64_t));
			w_dim=src_w_dim;
		}

		/** set bias
		 *
		 * @param b new bias
		 */
		inline void set_bias(float64_t b)
		{
			bias=b;
		}

		/** get bias
		 *
		 * @return bias
		 */
		inline float64_t get_bias()
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

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual inline void set_features(CStreamingDotFeatures* feat)
		{
			SG_UNREF(features);
			SG_REF(feat);
			features=feat;
		}

		/** apply linear machine to all examples
		 *
		 * @return resulting labels
		 */
		virtual CLabels* apply();

		/** apply linear machine to data
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data);

		/// get output for example "vec_idx"
		virtual float64_t apply(int32_t vec_idx)
		{
				SG_NOTIMPLEMENTED;
				return CMath::INFTY;
		}

		/**
		 * apply linear machine to one vector
		 *
		 * @param vec feature vector
		 * @param len length of vector
		 *
		 * @return classified label
		 */
		virtual float64_t apply(float64_t* vec, int32_t len);

		/**
		 * apply linear machine to vector currently being processed
		 *
		 * @return classified label
		 */
		virtual float64_t apply_to_this_example();

		/** get features
		 *
		 * @return features
		 */
		virtual CStreamingDotFeatures* get_features() { SG_REF(features); return features; }

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "OnlineLinearMachine"; }

	protected:
		/** dimension of w */
		int32_t w_dim;
		/** w */
		float64_t* w;
		/** bias */
		float64_t bias;
		/** features */
		CStreamingDotFeatures* features;
};
}
#endif
