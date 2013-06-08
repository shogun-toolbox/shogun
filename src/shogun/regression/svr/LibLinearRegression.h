/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef _REGRESSIONLIBLINEAR_H___
#define _REGRESSIONLIBLINEAR_H___
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/optimization/liblinear/shogun_liblinear.h>

namespace shogun
{
	/** liblinar regression solver type */
	enum LIBLINEAR_REGRESSION_TYPE
	{
		///L2 regularized support vector regression with L2 epsilon tube loss
		L2R_L2LOSS_SVR,
		///L2 regularized support vector regression with L1 epsilon tube loss
		L2R_L1LOSS_SVR_DUAL,
		///L2 regularized support vector regression with L2 epsilon tube loss (dual)
		L2R_L2LOSS_SVR_DUAL
	};

/** @brief LibLinear for regression
 */
class CLibLinearRegression : public CLinearMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_REGRESSION)

		/** default constructor  */
		CLibLinearRegression();

		/** standard constructor
		 * @param C C regularization constant value
		 * @param features features
		 * @param labs labels
		 */
		CLibLinearRegression(float64_t C, CDotFeatures* features, CLabels* labs);

		/** destructor */
		virtual ~CLibLinearRegression();

		/** returns regression type */
		inline LIBLINEAR_REGRESSION_TYPE get_liblinear_regression_type()
		{
			return m_liblinear_regression_type;
		}

		/** sets regression type */
		inline void set_liblinear_regression_type(LIBLINEAR_REGRESSION_TYPE st)
		{
			m_liblinear_regression_type=st;
		}

		/** get name */
		virtual const char* get_name() const
		{
			return "LibLinearRegression";
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

		/** set tube epsilon
		 *
		 * @param eps new tube epsilon
		 */
		inline void set_tube_epsilon(float64_t eps) { m_tube_epsilon=eps; }

		/** get tube epsilon
		 *
		 * @return tube epsilon
		 */
		inline float64_t get_tube_epsilon() { return m_tube_epsilon; }


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

protected:

		/** train machine */
		virtual bool train_machine(CFeatures* data = NULL);

private:
		/** solve svr with l1 or l2 loss */
		void solve_l2r_l1l2_svr(const liblinear_problem *prob);

		/** init defaults */
		void init_defaults();

		/** register parameters */
		void register_parameters();

protected:

		/** regularization constant for each machine */
		float64_t m_C;

		/** tolerance */
		float64_t m_epsilon;

		/** tube epsilon for support vector regression*/
		float64_t m_tube_epsilon;

		/** max number of iterations */
		int32_t m_max_iter;

		/** use bias */
		bool m_use_bias;

		/** which solver to use for regression */
		LIBLINEAR_REGRESSION_TYPE m_liblinear_regression_type;
};
}
#endif /* HAVE_LAPACK */
#endif
