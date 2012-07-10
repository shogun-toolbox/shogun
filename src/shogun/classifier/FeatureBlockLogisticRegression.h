/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  FEATUREBLOCKLOGISTICREGRESSION_H_
#define  FEATUREBLOCKLOGISTICREGRESSION_H_

#include <shogun/lib/config.h>
#include <shogun/lib/IndexBlockRelation.h>
#include <shogun/machine/SLEPMachine.h>

namespace shogun
{
/** @brief  */
class CFeatureBlockLogisticRegression : public CSLEPMachine
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CFeatureBlockLogisticRegression();

		/** constructor
		 *
		 * @param z regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 */
		CFeatureBlockLogisticRegression(
		     float64_t z, CDotFeatures* training_data, 
		     CBinaryLabels* training_labels, CIndexBlockRelation* task_relation);

		/** destructor */
		virtual ~CFeatureBlockLogisticRegression();

		/** get name */
		virtual const char* get_name() const 
		{
			return "FeatureBlockLogisticRegression";
		}

		/** getter for feature tree
		 * @return feature tree
		 */
		CIndexBlockRelation* get_feature_relation() const;

		/** setter for feature tree
		 * @param feature_tree feature tree
		 */
		void set_feature_relation(CIndexBlockRelation* feature_relation);

		virtual float64_t apply_one(int32_t vec_idx);

	protected:
		
		virtual SGVector<float64_t> apply_get_outputs(CFeatures* data);

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

	private:

		/** register parameters */
		void register_parameters();

	protected:

		/** feature tree */
		CIndexBlockRelation* m_feature_relation;

};
}
#endif
