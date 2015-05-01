/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#ifndef __SOSVM_HELPER_H__
#define __SOSVM_HELPER_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/StructuredModel.h>

namespace shogun
{

/** @brief class CSOSVMHelper contains helper functions to compute primal objectives,
 * dual objectives, average training losses, duality gaps etc. These values will be
 * recorded to check convergence. This class is inspired by the matlab implementation
 * of the block coordinate Frank-Wolfe SOSVM solver [1].
 *
 * [1] S. Lacoste-Julien, M. Jaggi, M. Schmidt and P. Pletscher. Block-Coordinate
 * Frank-Wolfe Optimization for Structural SVMs. ICML 2013.
 */
class CSOSVMHelper : public CSGObject
{
public:
	/** constructor */
	CSOSVMHelper();

	/** constructor
	 *
	 * @param bufsize size of buffer (default: 1000)
	 */
	CSOSVMHelper(int32_t bufsize);

	/** destructor */
	virtual ~CSOSVMHelper();

	/** @return name of SGSerializable */
	virtual const char* get_name() const { return "SOSVMHelper"; }

	/** Computes the primal objective of n-slack structured SVM
	 * \f$ \frac{\lambda}{2} \|w\|^2 + \frac{1}{n} \sum_i \max_y (L_i(y) - w^T\Psi_i(y)) \f$
	 *
	 * @param w is primal variables
	 * @param model structured model
	 * @param lbda regularization parameter lambda
	 * @return primal objective value
	 */
	static float64_t primal_objective(SGVector<float64_t> w, CStructuredModel* model, float64_t lbda);

	/** Computes the Lagrange dual objective of n-slack structured SVM
	 * Let \f$ Y_i \f$ denote the output space of sample i, and \f$ m = \sum_i |Y_i| \f$.
	 * The dual objective is \f$ -\frac{\lambda}{2} \|A\alpha\|^2 + b^T\alpha \f$,
	 * where \f$ A = \frac{1}{\lambda \cdot n}[\cdots, \psi_i(y), \cdots]_{d \times m} \f$,
	 * \f$ b = \frac{1}{n} [\cdots, L_i(y), \cdots]_{m} \f$,
	 * and \f$ \alpha \f$ are dual variables.
	 *
	 * @param w is primal variables, given by \f$ A\alpha \f$
	 * @param aloss is the average loss, given by \f$ b^T\alpha \f$
	 * @param lbda regularization parameter lambda
	 * @return dual objective value
	 */
	static float64_t dual_objective(SGVector<float64_t> w, float64_t aloss, float64_t lbda);

	/** Return the average loss for the predictions
	 *
	 * @param w is primal variables
	 * @param model structured model
	 * @param is_ub whether to compute the upper bound of average loss
	 * @return average loss
	 */
	static float64_t average_loss(SGVector<float64_t> w, CStructuredModel* model, bool is_ub = false);

	/** add debug information
	 *
	 * @param primal primal objective value
	 * @param eff_pass effective pass
	 * @param train_error training error
	 * @param dual dual objective value
	 * @param dgap duality gap
	 */
	virtual void add_debug_info(float64_t primal, float64_t eff_pass, float64_t train_error,
		float64_t dual = -1, float64_t dgap = -1);

	/** get primal objectives
	 *
	 * @return primal objectives
	 */
	SGVector<float64_t> get_primal_values() const;

	/** get dual objectives
	 *
	 * @return dual objectives
	 */
	SGVector<float64_t> get_dual_values() const;

	/** get duality gaps
	 *
	 * @return duality gaps
	 */
	SGVector<float64_t> get_duality_gaps() const;

	/** get effective passes
	 *
	 * @return effective passes
	 */
	SGVector<float64_t> get_eff_passes() const;

	/** get training errors
	 *
	 * @return training errors
	 */
	SGVector<float64_t> get_train_errors() const;

	/** terminate logging and resize vectors
	 */
	void terminate();

private:
	/** init parameters */
	void init();

private:
	/** history of primal value */
	SGVector<float64_t> m_primal;

	/** history of dual value */
	SGVector<float64_t> m_dual;

	/** history of duality gap */
	SGVector<float64_t> m_duality_gap;

	/** number of effective passes of data */
	SGVector<float64_t> m_eff_pass;

	/** history of training error */
	SGVector<float64_t> m_train_error;

	/** tracker of training progress */
	int32_t m_tracker;

	/** buffer size */
	int32_t m_bufsize;

}; /* CSOSVMHelper */

} /* namespace shogun */

#endif
