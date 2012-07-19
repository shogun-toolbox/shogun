/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef  MULTITASKCLUSTEREDLOGISTICREGRESSION_H_
#define  MULTITASKCLUSTEREDLOGISTICREGRESSION_H_

#include <shogun/transfer/multitask/MultitaskLogisticRegression.h>

namespace shogun
{
/** @brief  */
class CMultitaskClusteredLogisticRegression : public CMultitaskLogisticRegression
{

	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY)

		/** default constructor */
		CMultitaskClusteredLogisticRegression();

		/** constructor
		 *
		 * @param rho1 rho1 regularization coefficient
		 * @param rho2 rho2 regularization coefficient
		 * @param training_data training features
		 * @param training_labels training labels
		 * @param task_relation task relation
		 * @param num_cluster number of task clusters
		 */
		CMultitaskClusteredLogisticRegression(
		     float64_t rho1, float64_t rho2, CDotFeatures* training_data, 
		     CBinaryLabels* training_labels, CTaskGroup* task_group,
		     int32_t num_clusters);

		/** destructor */
		virtual ~CMultitaskClusteredLogisticRegression();

		/** get rho1
		 */
		int32_t get_rho1() const; 
		/** set rho1
		 * @param rho1 value
		 */
		void set_rho1(float64_t rho1); 
		/** get rho1
		 */
		int32_t get_rho2() const;
		/** set rho1
		 * @param rho2 value
		 */
		void set_rho2(float64_t rho2);
		/** get num clusters */
		int32_t get_num_clusters() const;
		/** set num clusters
		 * @param num_clusters number of clusters
		 */
		void set_num_clusters(int32_t num_clusters);

		/** get name */
		virtual const char* get_name() const 
		{
			return "MultitaskClusteredLogisticRegression";
		}

	protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data=NULL);

	protected:

		/** rho1 */
		float64_t m_rho1;
		
		/** rho2 */
		float64_t m_rho2;

		/** num clusters */
		int32_t m_num_clusters;
};
}
#endif
