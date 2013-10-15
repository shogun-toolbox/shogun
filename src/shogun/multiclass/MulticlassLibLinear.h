/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef _MULTICLASSLIBLINEAR_H___
#define _MULTICLASSLIBLINEAR_H___
#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/optimization/liblinear/shogun_liblinear.h>

namespace shogun
{

/** @brief multiclass LibLinear wrapper. Uses Crammer-Singer
    formulation and gradient descent optimization algorithm
    implemented in the LibLinear library. Regularized bias
    support is added using stacking bias 'feature' to
    hyperplanes normal vectors.

    In case of small changes of C or particularly epsilon
    this class provides ability to save whole liblinear
    training state (i.e. W vector, gradients, etc) and re-use
    on next train() calls. This capability could be
    enabled using set_save_train_state() method. Train
    state can be forced to clear using
    reset_train_state() method.
 */
class CMulticlassLibLinear : public CLinearMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor  */
		CMulticlassLibLinear();

		/** standard constructor
		 * @param C C regularization constant value
		 * @param features features
		 * @param labs labels
		 */
		CMulticlassLibLinear(float64_t C, CDotFeatures* features, CLabels* labs);

		/** destructor */
		virtual ~CMulticlassLibLinear();

		/** get name */
		virtual const char* get_name() const
		{
			return "MulticlassLibLinear";
		}

		/** set C
		 * @param C C value
		 */
		inline void set_C(float64_t C)
		{
			ASSERT(C>0)
			m_C = C;
		}
		/** get C
		 * @return C value
		 */
		inline float64_t get_C() const { return m_C; }

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

		/** set use bias
		 * @param use_bias use_bias value
		 */
		inline void set_use_bias(bool use_bias)
		{
			m_use_bias = use_bias;
		}
		/** get use bias
		 * @return use_bias value
		 */
		inline bool get_use_bias() const
		{
			return m_use_bias;
		}

		/** set save train state
		 * @param save_train_state save train state value
		 */
		inline void set_save_train_state(bool save_train_state)
		{
			m_save_train_state = save_train_state;
		}
		/** get save train state
		 * @return save_train_state value
		 */
		inline bool get_save_train_state() const
		{
			return m_save_train_state;
		}

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

		/** reset train state */
		void reset_train_state()
		{
			if (m_train_state)
			{
				delete m_train_state;
				m_train_state = NULL;
			}
		}

		/** get support vector indices
		 * @return support vector indices
		 */
		SGVector<int32_t> get_support_vectors() const;

protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data = NULL);

		/** obtain regularizer (w0) matrix */
		virtual SGMatrix<float64_t> obtain_regularizer_matrix() const;

private:

		/** init defaults */
		void init_defaults();

		/** register parameters */
		void register_parameters();

protected:

		/** regularization constant for each machine */
		float64_t m_C;

		/** tolerance */
		float64_t m_epsilon;

		/** max number of iterations */
		int32_t m_max_iter;

		/** use bias */
		bool m_use_bias;

		/** save train state */
		bool m_save_train_state;

		/** solver state */
		mcsvm_state* m_train_state;
};
}
#endif
