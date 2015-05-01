/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Shell Hu
 * Copyright (C) 2014 Shell Hu
 */

#ifndef __FW_SOSVM_H__
#define __FW_SOSVM_H__

#include <shogun/lib/config.h>

#include <shogun/machine/LinearStructuredOutputMachine.h>

namespace shogun
{

/** @brief Class CFWSOSVM solves SOSVM using Frank-Wolfe algorithm [1].
 *
 * [1] S. Lacoste-Julien, M. Jaggi, M. Schmidt and P. Pletscher. Block-Coordinate
 * Frank-Wolfe Optimization for Structural SVMs. ICML 2013.
 */
class CFWSOSVM : public CLinearStructuredOutputMachine
{
public:
	/** default constructor */
	CFWSOSVM();

	/** standard constructor
	 *
	 * @param model structured model with application specific functions
	 * @param labs structured labels
	 * @param do_line_search whether do analytical line search
	 * @param verbose whether compute debug information, such as primal value, duality gap etc.
	 */
	CFWSOSVM(CStructuredModel* model, CStructuredLabels* labs,
			bool do_line_search = true, bool verbose = false);

	/** destructor */
	~CFWSOSVM();

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "FWSOSVM"; }

	/** get classifier type
	 *
	 * @return classifier type CT_FWSOSVM
	 */
	virtual EMachineType get_classifier_type();

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

	/** @return threshold of the duality gap */
	float64_t get_gap_threshold() const;

	/** set threshold of the duality gap
	 *
	 * @param gap_threshold threshold of the duality gap
	 */
	void set_gap_threshold(float64_t gap_threshold);

	/** @return the average loss ell */
	float64_t get_ell() const;

	/** set the average loss ell
	 *
	 * @param ell the average loss
	 */
	void set_ell(float64_t ell);

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
	bool m_do_line_search;

	/** Stop threshold of the duality gap (default: 0.1) */
	float64_t m_gap_threshold;

	/** Average loss */
	float64_t m_ell;

}; /* CFWSOSVM */

} /* namespace shogun */

#endif
