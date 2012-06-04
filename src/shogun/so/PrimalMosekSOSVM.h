/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _PRIMAL_MOSEK_SOSVM__H__
#define _PRIMAL_MOSEK_SOSVM__H__

#ifdef USE_MOSEK

#define DEBUG_PRIMAL_MOSEK_SOSVM

#include <shogun/machine/LinearStructuredOutputMachine.h>

namespace shogun
{

/** 
 * @brief Class PrimalMosekSOSVM that implements the optimization
 * algorithm for structured output (SO) problems presented in [1] for SVM
 * learning. The optimization problem is solved using a cutting plane algorithm.
 *
 * [1] Tsochantaridis, I., Hofmann, T., Joachims, T., Altun, Y.
 *     Support Vector Machine Learning for Interdependent and Structured Ouput
 *     Spaces.
 *     http://www.cs.cornell.edu/People/tj/publications/tsochantaridis_etal_04a.pdf
 */
class CPrimalMosekSOSVM : public CLinearStructuredOutputMachine
{
	public:
		/** default constructor */
		CPrimalMosekSOSVM();

		/** standard constructor
		 *
		 * @param model structured model with application specific functions
		 * @param loss structured loss function
		 * @param labs structured labels
		 * @param features features
		 */
		CPrimalMosekSOSVM(CStructuredModel* model, CLossFunction* loss, CStructuredLabels* labs, CFeatures* features);

		/** destructor */
		~CPrimalMosekSOSVM();

	protected:
		/** train primal SO-SVM
		 *
		 * @param data training data
		 * @return whether the training was successful
		 */
		bool train_machine(CFeatures* data = NULL);

	private:
		/** register class members */
		void register_parameters();

		/** computes the result of TODO equation
		 *
		 * @param result CResultSet structure with any argmax output
		 * @return result of the operation
		 */
		float64_t compute_loss_arg(CResultSet* result) const;

		/** insert element in the list of argmax results
		 *
		 * @param result_list list of CResultSet
		 * @param result element to insert in the list
		 */
		bool insert_result(CList* result_list, CResultSet* result) const;

		/** TODO doc
		 *
		 * @param lb lower bound for 
		 */
		bool solve_qp(SGMatrix< float64_t > A, SGMatrix< float64_t > C,
				SGVector< float64_t > lb, 
				SGVector< float64_t > ub) const;

		/** TODO doc
		 *
		 */
		double predicted_delta_loss(int32_t idx) const;

	private:
		/** weight vector */
		SGVector< float64_t > m_w;

}; /* class CPrimalMosekSOSVM */

} /* namespace shogun */

#endif /* USE_MOSEK */
#endif /* _PRIMAL_MOSEK_SOSVM__H__ */
