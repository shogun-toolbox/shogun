/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LINEARCLASSIFIER_H__
#define _LINEARCLASSIFIER_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
#include <shogun/lib/SGVector.h>


namespace shogun
{

class CBinaryLabels;
class CDotFeatures;
class CFeatures;
class CRegressionLabels;

/** @brief Class LinearMachine is a generic interface for all kinds of linear
 * machines like classifiers.
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
 * Note that this framework works with linear classifiers of arbitraty feature
 * type, e.g. dense and sparse and even string based features. This is
 * implemented by using CDotFeatures that may provide a mapping function
 * \f$\Phi({\bf x})\mapsto {\cal R^D}\f$ encapsulating all the required
 * operations (like the dot product). The decision function is thus
 *
 *  \f[
 *		f({\bf x})= {\bf w} \cdot \Phi({\bf x}) + b.
 *	\f]
 *
 *	The following linear classifiers are implemented
 *	\li Linear Descriminant Analysis (CLDA)
 *	\li Linear Programming Machines (CLPM, CLPBoost)
 *	\li Perceptron (CPerceptron)
 *	\li Linear SVMs (CSVMSGD, CLibLinear, CSVMOcas, CSVMLin, CSubgradientSVM)
 *
 *	\sa CDotFeatures
 *
 * */
class CLinearMachine : public CMachine
{
	public:
		/** default constructor */
		CLinearMachine();

		/** Constructor
		 *
		 * @param  compute_bias new m_compute_bias
		 * Determines if bias_compution is considered or not
		 */
		CLinearMachine(bool compute_bias);

		/** destructor */
		virtual ~CLinearMachine();

		/** copy constructor */
		CLinearMachine(CLinearMachine* machine);

		/** Train machine
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get w
		 *
		 * @return weight vector
		 */
		virtual SGVector<float64_t> get_w() const;

		/** set w
		 *
		 * @param src_w new w
		 */
		virtual void set_w(const SGVector<float64_t> src_w);

		/** set bias
		 *
		 * @param b new bias
		 */
		virtual void set_bias(float64_t b);

		/** get bias
		 *
		 * @return bias
		 */
		virtual float64_t get_bias();

		/** Set m_compute_bias
		 *
		 * Determines if bias compution is considered or not
		 * @param  compute_bias new m_compute_bias
		 */
		virtual void set_compute_bias(bool compute_bias);

		/** Get compute bias
		 *
		 * @return compute_bias
		 */
		virtual bool get_compute_bias();		

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(CDotFeatures* feat);

		/** apply linear machine to data
		 * for binary classification problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CBinaryLabels* apply_binary(CFeatures* data=NULL);

		/** apply linear machine to data
		 * for regression problem
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

		/** applies to one vector */
		virtual float64_t apply_one(int32_t vec_idx);

		/** get features
		 *
		 * @return features
		 */
		virtual CDotFeatures* get_features();

		/** Returns the name of the SGSerializable instance.  It MUST BE
		 *  the CLASS NAME without the prefixed `C'.
		 *
		 * @return name of the SGSerializable
		 */
		virtual const char* get_name() const { return "LinearMachine"; }

	protected:

		/** apply get outputs
		 *
		 * @param data features to compute outputs
		 * @return outputs
		 */
		virtual SGVector<float64_t> apply_get_outputs(CFeatures* data);

		/** Stores feature data of underlying model. Does nothing because
		 * Linear machines store the normal vector of the separating hyperplane
		 * and therefore the model anyway
		 */
		virtual void store_model_features();

		/** Computes the added bias. The bias is computed 
		 * as the mean error between the predictions and 
		 * the true labels.
		*/
		void compute_bias(CFeatures* data);

	private:

		void init();

	private:

		/** w */
		SGVector<float64_t> w;

	protected:
		/** bias */
		float64_t bias;
		/** features */
		CDotFeatures* features;
		/** If true, bias is computed in train method */
		bool m_compute_bias;
};
}
#endif
