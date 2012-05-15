/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  FEATURETREELEASTSQUARESREGRESSION_H_
#define  FEATURETREELEASTSQUARESREGRESSION_H_

#include <shogun/lib/config.h>
#include <shogun/transfer/Trees.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
/** @brief  */
class CFeatureTreeLeastSquaresRegression : public CLinearMachine
{
	public:
		/** default constructor */
		CFeatureTreeLeastSquaresRegression();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param feature_tree feature tree 
		 */
		CFeatureTreeLeastSquaresRegression(
		     float64_t z, CDotFeatures* training_data, 
		     CLabels* training_labels, CFeatureTree* feature_tree);

		/** destructor */
		virtual ~CFeatureTreeLeastSquaresRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "FeatureTreeLeastSquaresRegression";
		}

		CFeatureTree* get_feature_tree() const;
		int32_t get_max_iter() const;
		int32_t get_regularization() const;
		int32_t get_termination() const;
		float64_t get_tolerance() const;
		float64_t get_z() const;

		void set_feature_tree(CFeatureTree* feature_tree);
		void set_max_iter(int32_t max_iter);
		void set_regularization(int32_t regularization);
		void set_termination(int32_t termination);
		void set_tolerance(float64_t tolerance);
		void set_z(float64_t z);

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		int32_t m_regularization;

		int32_t m_termination;

		int32_t m_max_iter;

		float64_t m_tolerance;

		/** regularization coefficient */
		float64_t m_z;

		/** feature tree */
		CFeatureTree* m_feature_tree;

};
}
#endif   /* ----- #ifndef FEATURETREELEASTSQUARESREGRESSION_H_  ----- */
