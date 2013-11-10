/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#ifndef __STOCHASTIC_SOSVM_H__
#define __STOCHASTIC_SOSVM_H__

#include <shogun/lib/SGVector.h>
#include <shogun/machine/LinearStructuredOutputMachine.h>

namespace shogun
{

/** @brief Class CStochasticSOSVM solves SOSVM using stochastic subgradient descent
 * on the SVM primal problem [1], which is equivalent to SGD or Pegasos [2].
 * This class is inspired by the matlab SGD implementation in [3].
 *
 * [1] N. Ratliff, J. A. Bagnell, and M. Zinkevich. (online) subgradient methods
 * for structured prediction. AISTATS, 2007.
 * [2] S. Shalev-Shwartz, Y. Singer, N. Srebro. Pegasos: Primal Estimated 
 * sub-GrAdient SOlver for SVM. ICML 2007.
 * [3] S. Lacoste-Julien, M. Jaggi, M. Schmidt and P. Pletscher. Block-Coordinate 
 * Frank-Wolfe Optimization for Structural SVMs. ICML 2013.
 */
class CStochasticSOSVM : public CLinearStructuredOutputMachine
{
public:
	/** default constructor */
	CStochasticSOSVM();

	/** standard constructor
	 *
	 * @param model structured model with application specific functions
	 * @param labs structured labels
	 * @param do_weighted_averaging whether mix w with previous average weights
	 * @param verbose whether compute debug information, such as primal value, duality gap etc.
	 */
	CStochasticSOSVM(CStructuredModel* model, CStructuredLabels* labs, 
		bool do_weighted_averaging = true, bool verbose = false);

	/** destructor */
	~CStochasticSOSVM();

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "StochasticSOSVM"; }

	/** get classifier type
	 *
	 * @return classifier type CT_STOCHASTICSOSVM
	 */
	virtual EMachineType get_classifier_type() const;

	/** @return lambda */
	float64_t get_lambda() const;
	
	/** set regularization const
	 *
	 * @param lbda regularization const lambda
	 */
	void set_lambda(float64_t lbda);
	
	/** @return number of iterations */
	int32_t get_num_iter() const;
	
	/** set max number of iterations
	 *
	 * @param num_iter number of iterations
	 */
	void set_num_iter(int32_t num_iter);
	
	/** @return debug multiplier */
	int32_t get_debug_multiplier() const;
	
	/** set frequency of debug outputs
	 *
	 * @param multiplier debug multiplier
	 */
	void set_debug_multiplier(int32_t multiplier);
	
	/** @return rand seed */
	uint32_t get_rand_seed() const;
	
	/** set random seed
	 *
	 * @param rand_seed random seed
	 */
	void set_rand_seed(uint32_t rand_seed);

protected:
	/** train primal SO-SVM
	 *
	 * @param data training data
	 * @return whether the training was successful
	 */
	virtual bool train_machine(CFeatures* data = NULL);

private:
	/** register and initialize parameters */
	void init();

private:
	/** The regularization constant (default: 1/n) */
	float64_t m_lambda;

	/** Number of passes through the data (default: 50) */
	int32_t m_num_iter;

	/** Whether to use weighted averaging of the iterates */
	bool m_do_weighted_averaging;

	/** random seed */
	uint32_t m_rand_seed;

	/** If set to 0, the algorithm computes the objective after each full
	 * pass trough the data. If in (0,100) logging happens at a
	 * geometrically increasing sequence of iterates, thus allowing for
	 * within-iteration logging. The smaller the number, the more
	 * costly the computations will be!
	 */
	int32_t m_debug_multiplier;

}; /* CStochasticSOSVM */

} /* namespace shogun */

#endif
