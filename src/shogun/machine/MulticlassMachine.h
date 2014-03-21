/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2011 Soeren Sonnenburg
 * Written (W) 2012 Fernando José Iglesias García and Sergey Lisitsyn
 * Written (W) 2013 Shell Hu and Heiko Strathmann
 * Copyright (C) 2012 Sergey Lisitsyn, Fernando José Iglesias Garcia
 */

#ifndef _MULTICLASSMACHINE_H___
#define _MULTICLASSMACHINE_H___

#include <shogun/lib/config.h>

#include <shogun/machine/BaseMulticlassMachine.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/multiclass/MulticlassStrategy.h>

namespace shogun
{

class CFeatures;
class CLabels;
class CMulticlassLabels;
class CMulticlassMultipleOutputLabels;

/** @brief experimental abstract generic multiclass machine class */
class CMulticlassMachine : public CBaseMulticlassMachine
{
	public:
		/** default constructor  */
		CMulticlassMachine();

		/** standard constructor
		 * @param strategy multiclass strategy
		 * @param machine machine
		 * @param labels labels
		 */
		CMulticlassMachine(CMulticlassStrategy* strategy, CMachine* machine, CLabels* labels);

		/** destructor */
		virtual ~CMulticlassMachine();

		/** set labels
		 *
		 * @param lab labels
		 */
		virtual void set_labels(CLabels* lab);

		/** set machine
		 *
		 * @param num index of machine
		 * @param machine machine to set
		 * @return if setting was successful
		 */
		inline bool set_machine(int32_t num, CMachine* machine)
		{
			ASSERT(num<m_machines->get_num_elements() && num>=0)
			if (machine != NULL && !is_acceptable_machine(machine))
				SG_ERROR("Machine %s is not acceptable by %s", machine->get_name(), this->get_name())

			m_machines->set_element(machine, num);
			return true;
		}

		/** get machine
		 *
		 * @param num index of machine to get
		 * @return SVM at number num
		 */
		inline CMachine* get_machine(int32_t num) const
		{
			return (CMachine*) m_machines->get_element_safe(num);
		}

		/** get outputs of i-th submachine
		 * @param i number of submachine
		 * @return outputs
		 */
		virtual CBinaryLabels* get_submachine_outputs(int32_t i);

		/** get output of i-th submachine for num-th vector
		 * @param i number of submachine
		 * @param num number of feature vector
		 * @return output
		 */
		virtual float64_t get_submachine_output(int32_t i, int32_t num);

		/** classify all examples
		 *
		 * @return resulting labels
		 */
		virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

		/** classify all examples with multiple output
		 *
		 * @return resulting labels
		 */
		virtual CMulticlassMultipleOutputLabels* apply_multiclass_multiple_output(CFeatures* data=NULL, int32_t n_outputs=5);

		/** classify one example
		 * @param vec_idx
		 * @return label
		 */
		virtual float64_t apply_one(int32_t vec_idx);

		/** get the type of multiclass'ness
		 *
		 * @return multiclass type one vs one etc
		 */
		inline CMulticlassStrategy* get_multiclass_strategy() const
		{
			SG_REF(m_multiclass_strategy);
			return m_multiclass_strategy;
		}

		/** returns rejection strategy
		 *
		 * @return rejection strategy
		 */
		inline CRejectionStrategy* get_rejection_strategy() const
		{
			return m_multiclass_strategy->get_rejection_strategy();
		}

		/** sets rejection strategy
		 *
		 * @param rejection_strategy rejection strategy to be set
		 */
		inline void set_rejection_strategy(CRejectionStrategy* rejection_strategy)
		{
			m_multiclass_strategy->set_rejection_strategy(rejection_strategy);
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "MulticlassMachine";
		}

		/** get prob output heuristic of multiclass strategy */
		inline EProbHeuristicType get_prob_heuris()
		{
			return m_multiclass_strategy->get_prob_heuris_type();
		}

		/** set prob output heuristic of multiclass strategy
		 * @param prob_heuris type of probability heuristic
		 */
		inline void set_prob_heuris(EProbHeuristicType prob_heuris)
		{
			m_multiclass_strategy->set_prob_heuris_type(prob_heuris);
		}

	protected:
		/** init strategy */
		void init_strategy();

		/** clear machines */
		void clear_machines();

		/** train machine */
		virtual bool train_machine(CFeatures* data = NULL);

		/** abstract init machine for training method */
		virtual bool init_machine_for_train(CFeatures* data) = 0;

		/** abstract init machines for applying method */
		virtual bool init_machines_for_apply(CFeatures* data) = 0;

		/** check whether machine is ready */
		virtual bool is_ready() = 0;

		/** obtain machine from trained one */
		virtual CMachine* get_machine_from_trained(CMachine* machine) = 0;

		/** get num rhs vectors */
		virtual int32_t get_num_rhs_vectors() = 0;

		/** set subset to the features of the machine, deletes old one
		 *
		 * @param subset subset indices to set
		 */
		virtual void add_machine_subset(SGVector<index_t> subset) = 0;

		/** deletes any subset set to the features of the machine */
		virtual void remove_machine_subset() = 0;

		/** whether the machine is acceptable in set_machine */
		virtual bool is_acceptable_machine(CMachine *machine)
		{
			return true;
		}

	private:

		/** register parameters */
		void register_parameters();

	protected:
		/** type of multiclass strategy */
		CMulticlassStrategy *m_multiclass_strategy;

		/** machine */
		CMachine* m_machine;
};
}
#endif
