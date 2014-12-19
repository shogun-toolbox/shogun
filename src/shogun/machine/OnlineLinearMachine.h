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

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/machine/Machine.h>


namespace shogun
{
class CBinaryLabels;
class CFeatures;
class CRegressionLabels;

/** @brief Class OnlineLinearMachine is a generic interface for linear
 * machines like classifiers which work through online algorithms.
 *
 * A linear classifier computes
 *
 *  \f[
 *		f({\bf x})= {\bf w} \cdot {\bf x} + b
 *	\f]
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
 *		f({\bf x})= {\bf w} \cdot \Phi({\bf x}) + b.
 *	\f]
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
		virtual void get_w(float32_t*& dst_w, int32_t& dst_dims)
		{
			ASSERT(w && w_dim>0)
			dst_w=w;
			dst_dims=w_dim;
		}

		/**
		 * Get w as a _new_ float64_t array
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		virtual void get_w(float64_t*& dst_w, int32_t& dst_dims)
		{
			ASSERT(w && w_dim>0)
			dst_w=SG_MALLOC(float64_t, w_dim);
			for (int32_t i=0; i<w_dim; i++)
				dst_w[i]=w[i];
			dst_dims=w_dim;
		}

		/** get w
		 *
		 * @return weight vector
		 */
		virtual SGVector<float32_t> get_w()
		{
			float32_t * dst_w = SG_MALLOC(float32_t, w_dim);
			for (int32_t i=0; i<w_dim; i++)
				dst_w[i]=w[i];
			return SGVector<float32_t>(dst_w, w_dim);
		}

		/** set w
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w
		 */
		virtual void set_w(float32_t* src_w, int32_t src_w_dim)
		{
			SG_FREE(w);
			w=SG_MALLOC(float32_t, src_w_dim);
			memcpy(w, src_w, size_t(src_w_dim)*sizeof(float32_t));
			w_dim=src_w_dim;
		}

		/**
		 * Set weight vector from a float64_t vector
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w
		 */
		virtual void set_w(float64_t* src_w, int32_t src_w_dim)
		{
			SG_FREE(w);
			w=SG_MALLOC(float32_t, src_w_dim);
			for (int32_t i=0; i<src_w_dim; i++)
				w[i] = src_w[i];
			w_dim=src_w_dim;
		}

		/** set bias
		 *
		 * @param b new bias
		 */
		virtual void set_bias(float32_t b)
		{
			bias=b;
		}

		/** get bias
		 *
		 * @return bias
		 */
		virtual float32_t get_bias()
		{
			return bias;
		}

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(CStreamingDotFeatures* feat)
		{
			SG_REF(feat);
			SG_UNREF(features);
			features=feat;
		}

		/** apply linear machine to data
		 * for regression problems
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

		/** apply linear machine to data
		 * for binary classification problems
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

		/// get output for example "vec_idx"
		virtual float64_t apply_one(int32_t vec_idx)
		{
			SG_NOTIMPLEMENTED
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
		virtual float32_t apply_one(float32_t* vec, int32_t len);

		/**
		 * apply linear machine to vector currently being processed
		 *
		 * @return classified label
		 */
		virtual float32_t apply_to_current_example();

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

		/** Start training of the online machine, sub-class should override
		 * this if some preparations are to be done
		 */
		virtual void start_train() { }

		/** Stop training of the online machine, sub-class should override
		 * this if some clean up is needed
		 */
		virtual void stop_train() { }

		/** train on one example
		 * @param feature the feature object containing the current example. Note that get_next_example
		 *        is already called so relevalent methods like dot() and dense_dot() can be directly
		 *        called. WARN: this function should only process ONE example, and get_next_example()
		 *        should NEVER be called here. Use the label passed in the 2nd parameter, instead of
		 *		  get_label() from feature, because sometimes the features might not have associated
		 *		  labels or the caller might want to provide some other labels.
		 * @param label label of this example
		 */
		virtual void train_example(CStreamingDotFeatures *feature, float64_t label) { SG_NOTIMPLEMENTED }

	protected:
		/**
		 * Train classifier
		 *
		 * @param data Training data, can be avoided if already
		 * initialized with it
		 *
		 * @return Whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

		/** get real outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		SGVector<float64_t> apply_get_outputs(CFeatures* data);

		/** whether train require labels */
		virtual bool train_require_labels() const { return false; }

	protected:
		/** dimension of w */
		int32_t w_dim;
		/** w */
		float32_t* w;
		/** bias */
		float32_t bias;
		/** features */
		CStreamingDotFeatures* features;
};
}
#endif
