/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef MULTICLASSTREEGUIDEDLOGISTICREGRESSION_H_
#define MULTICLASSTREEGUIDEDLOGISTICREGRESSION_H_
#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/lib/IndexBlockTree.h>

namespace shogun
{

/** @brief multiclass tree guided logistic regression
 */
class CMulticlassTreeGuidedLogisticRegression : public CLinearMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor  */
		CMulticlassTreeGuidedLogisticRegression();

		/** standard constructor
		 * @param z z regularization constant value
		 * @param feats features
		 * @param labs labels
		 * @param tree tree
		 */
		CMulticlassTreeGuidedLogisticRegression(float64_t z, CDotFeatures* feats, CLabels* labs, CIndexBlockTree* tree);

		/** destructor */
		virtual ~CMulticlassTreeGuidedLogisticRegression();

		/** get name */
		virtual const char* get_name() const
		{
			return "MulticlassTreeGuidedLogisticRegression";
		}

		/** set z
		 * @param z z value
		 */
		inline void set_z(float64_t z)
		{
			ASSERT(z>0)
			m_z = z;
		}
		/** get C
		 * @return C value
		 */
		inline float64_t get_z() const { return m_z; }

		/** set epsilon
		 * @param epsilon epsilon value
		 */
		inline void set_epsilon(float64_t epsilon)
		{
			ASSERT(epsilon>0)
			m_epsilon = epsilon;
		}
		/** get epsilon
		 * @return epsilon value
		 */
		inline float64_t get_epsilon() const { return m_epsilon; }

		/** set max iter
		 * @param max_iter max iter value
		 */
		inline void set_max_iter(int32_t max_iter)
		{
			ASSERT(max_iter>0)
			m_max_iter = max_iter;
		}
		/** get max iter
		 * @return max iter value
		 */
		inline int32_t get_max_iter() const { return m_max_iter; }

		/** set index tree
		 * @param index_tree index tree
		 */
		inline void set_index_tree(CIndexBlockTree* index_tree)
		{
			SG_REF(index_tree);
			SG_UNREF(m_index_tree);
			m_index_tree = index_tree;
		}
		/** get index tree
		 * @return current index tree
		 */
		inline CIndexBlockTree* get_index_tree() const
		{
			SG_REF(m_index_tree);
			return m_index_tree;
		}

protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data = NULL);

private:

		/** init defaults */
		void init_defaults();

		/** register parameters */
		void register_parameters();

protected:

		/** index tree */
		CIndexBlockTree* m_index_tree;

		/** regularization constant for each machine */
		float64_t m_z;

		/** tolerance */
		float64_t m_epsilon;

		/** max number of iterations */
		int32_t m_max_iter;

};
}
#endif /* HAVE_EIGEN3 */
#endif
