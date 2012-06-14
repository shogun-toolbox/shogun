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
#include <shogun/lib/IndicesTree.h>
#include <shogun/machine/SLEPMachine.h>

namespace shogun
{
/** @brief  */
class CFeatureTreeLeastSquaresRegression : public CSLEPMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_REGRESSION)

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
		     CRegressionLabels* training_labels, CIndicesTree* feature_tree);

		/** destructor */
		virtual ~CFeatureTreeLeastSquaresRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "FeatureTreeLeastSquaresRegression";
		}

		CIndicesTree* get_feature_tree() const;
		void set_feature_tree(CIndicesTree* feature_tree);

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		/** feature tree */
		CIndicesTree* m_feature_tree;

};
}
#endif   /* ----- #ifndef FEATURETREELEASTSQUARESREGRESSION_H_  ----- */
