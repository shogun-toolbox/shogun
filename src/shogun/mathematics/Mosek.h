/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _CMOSEK__H__
#define _CMOSEK__H__

#ifdef USE_MOSEK

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

#include <mosek.h>

namespace shogun
{

/** @brief Class CMosek to encapsulate access to the commercial MOSEK
 * purpose optimizer.
 *
 * This class provides methods to set up optimization problems that are
 * used in shogun, e.g. from PrimalMosekSOSVM.
 */
class CMosek : public CSGObject
{

	public:
		/** default constructor */
		CMosek();

		/**
		 * constructor to initialize the environment and an
		 * optimization task. It accepts two arguments to use
		 * as estimates on the maximum number of constraints in the
		 * task. The use of these arguments may speed up inputting,
		 * although it is not necessary. If these values are unknown,
		 * set them to zero.
		 *
		 * @param num_con estimate on the maximum number of constraints
		 * @param num_var estimate on the maximum number of variables
		 */
		CMosek(int32_t num_con, int32_t num_var);

		/** destructor */
		~CMosek();

		/** get rescode
		 *
		 * @return rescode result code of internal MOSEK functions
		 */
		inline MSKrescodee get_rescode() const { return m_rescode; }

		/**
		 * method used to direct the log stream of MOSEK
		 * functions to SG_PRINT
		 *
		 * @param handle function handler
		 * @param str string to print on screen
		 */
		static void MSKAPI print(void* handle, char str[]);

		/**
		 * initialize some of the terms for the optimization
		 * problem to solve in SO-SVM. These terms are basically
		 * those which does not change in the main loop of the
		 * cutting plane algorithm.
		 *
		 * @param M dimensionality of the joint feature space
		 * @param N number of training examples
		 * @param C regularization matrix for the optimization vector
		 * @param lb lower bounds for the optimization vector
		 * @param ub upper bounds for the optimization vector
		 * @param A constraints matrix
		 * @param b upper bounds for the constraints
		 *
		 * @return MSK result code
		 */
		MSKrescodee init_sosvm(int32_t M, int32_t N,
				int32_t num_aux, int32_t num_aux_con,
				SGMatrix< float64_t > C, SGVector< float64_t > lb,
				SGVector< float64_t > ub, SGMatrix< float64_t > A,
				SGVector< float64_t > b);

		/**
		 * adds a constraint to the MOSEK optimization task of
		 * the type used in SO-SVM.
		 *
		 * @param dPsi leftmost part of the constraint
		 * @param con_idx row index in A of the constraint
		 * @param train_idx index of the training example
		 * associated to the new constraint
		 * @param bi upper bound of the constraint
		 *
		 * @return MSK result code
		 */
		MSKrescodee add_constraint_sosvm(SGVector< float64_t > dPsi,
				index_t con_idx, index_t train_idx, int32_t num_aux,
				float64_t bi);

		/**
		 * wrapper for MOSEK's function MSK_putaveclist used
		 * to set the values in the linear constraint matrix A
		 *
		 * @param task an optimization task
		 * @param A new linear constraint matrix
		 *
		 * @return MSK result code
		 */
		static MSKrescodee wrapper_putaveclist(MSKtask_t & task, SGMatrix< float64_t > A);

		/**
		 * wrapper for MOSEK's function MSK_putboundlist used
		 * to set the bounds for either some constraints or variables
		 *
		 * @param task an optimization task
		 * @param b vector with bounds
		 *
		 * @return MSK result code
		 */
		static MSKrescodee wrapper_putboundlist(MSKtask_t & task, SGVector< float64_t > b);

		/**
		 * wrapper for MOSEK's function MSK_putqobj used to
		 * set the values in the regularization matrix of the
		 * quadratic objective term
		 *
		 * @param Q0 new regularization matrix, assumed to be
		 * symmetric
		 *
		 * @return MSK result code
		 */
		MSKrescodee wrapper_putqobj(SGMatrix< float64_t > Q0) const;

		/** solve the optimization task member
		 *
		 * @param sol where to write the optimization vector
		 *
		 * @return MSK result code
		 */
		MSKrescodee optimize(SGVector< float64_t > sol);

		/** free resources associated to MOSEK task and environment */
		void delete_problem();

		/**
		 * prints the terms involved in the problem that have been given
		 * to the task. Currently this method shows the terms that appear
		 * in the QP that takes the form
		 *
		 * min_x 0.5*x'*Q^0*x + c'*x
		 * s.t. A*x <= b, lb <= x <= ub
		 */
		void display_problem();

		/**
		 * Obtains the primal objective value for a solution.
		 *
		 * @return the primal objective value.
		 */
		float64_t get_primal_objective_value() const;

		/** @return object name */
		virtual const char* get_name() const { return "Mosek"; }

	private:
		/** MOSEK environment */
		MSKenv_t m_env;

		/** MOSEK optimization task */
		MSKtask_t m_task;

		/** MOSEK response code */
		MSKrescodee m_rescode;
};

} /* namespace shogun */

#endif /* USE_MOSEK */
#endif /* _CMOSEK__H__ */
