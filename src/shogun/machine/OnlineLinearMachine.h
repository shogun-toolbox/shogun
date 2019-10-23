/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Shashwat Lal Das, Thoralf Klein, 
 *          Evgeniy Andreev, Yuyu Zhang, Chiyuan Zhang, Viktor Gal, Weijie Lin, 
 *          Bjoern Esser, Saurabh Goyal
 */

#ifndef _ONLINELINEARCLASSIFIER_H__
#define _ONLINELINEARCLASSIFIER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/machine/Machine.h>


namespace shogun
{
class BinaryLabels;
class Features;
class RegressionLabels;

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
class OnlineLinearMachine : public Machine
{
	public:
		/** default constructor */
		OnlineLinearMachine();
		virtual ~OnlineLinearMachine();

		/**
		 * Get w as a _new_ float64_t array
		 *
		 * @param dst_w store w in this argument
		 * @param dst_dims dimension of w
		 */
		virtual void get_w(float64_t*& dst_w, int32_t& dst_dims)
		{
			ASSERT(m_w.vector && m_w.vlen > 0)
			dst_w=SG_MALLOC(float64_t, m_w.vlen);
			for (int32_t i=0; i<m_w.vlen; i++)
				dst_w[i]=m_w[i];
			dst_dims=m_w.vlen;
		}

		/** get w
		 *
		 * @return weight vector
		 */
		virtual SGVector<float32_t> get_w() const
		{
			return m_w;
		}

		/** set w
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w
		 */
		virtual void set_w(const SGVector<float32_t> w)
		{
			m_w = w;
		}

		/**
		 * Set weight vector from a float64_t vector
		 *
		 * @param src_w new w
		 * @param src_w_dim dimension of new w
		 */
		virtual void set_w(float64_t* src_w, int32_t src_w_dim)
		{
			m_w = SGVector<float32_t>(src_w_dim);
			for (int32_t i=0; i<src_w_dim; i++)
				m_w[i] = src_w[i];
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
		virtual void set_features(std::shared_ptr<StreamingDotFeatures> feat)
		{
			
			
			features=feat;
		}

		/** apply linear machine to data
		 * for regression problems
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL);

		/** apply linear machine to data
		 * for binary classification problems
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL);

		/// get output for example "vec_idx"
		virtual float64_t apply_one(int32_t vec_idx)
		{
			not_implemented(SOURCE_LOCATION);
			return Math::INFTY;
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
		virtual std::shared_ptr<StreamingDotFeatures> get_features() {  return features; }

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
		virtual void train_example(std::shared_ptr<StreamingDotFeatures >feature, float64_t label) { not_implemented(SOURCE_LOCATION); }

		/** whether train require labels */
		virtual bool train_require_labels() const
		{
			return false;
		}

	protected:
		/**
		 * Train classifier
		 *
		 * @param data Training data, can be avoided if already
		 * initialized with it
		 *
		 * @return Whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data=NULL);

		/** get real outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		SGVector<float64_t> apply_get_outputs(const std::shared_ptr<Features>& data);

	protected:
		/** w */
		SGVector<float32_t> m_w;
		/** bias */
		float32_t bias;
		/** features */
		std::shared_ptr<StreamingDotFeatures> features;
};
}
#endif
