/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef LEASTANGLEREGRESSION_H__
#define LEASTANGLEREGRESSION_H__

#include <lib/config.h>

#ifdef HAVE_LAPACK
#include <vector>
#include <machine/LinearMachine.h>

namespace shogun
{
class CFeatures;

/** @brief Class for Least Angle Regression, can be used to solve LASSO.
 *
 * LASSO is basically L1 regulairzed least square regression
 *
 * \f[
 * \min \|X^T\beta - y\|^2 + \lambda\|\beta\|_{1}
 * \f]
 *
 * where the L1 norm is defined as
 *
 * \f[
 * \|\beta\|_1 = \sum_i|\beta_i|
 * \f]
 *
 * **Note**: pre-processing of X and y are needed to ensure the correctness
 * of this algorithm:
 * * X needs to be normalized: each feature should have zero-mean and unit-norm
 * * y needs to be centered: its mean should be zero
 *
 * The above equation is equivalent to the following form
 *
 * \f[
 * \min \|X^T\beta - y\|^2 \quad s.t. \|\beta\|_1 \leq C
 * \f]
 *
 * There is a correspondence between the regularization coefficient lambda
 * and the hard constraint constant C. The latter form is easier to control
 * by explicitly constraining the l1-norm of the estimator. In this
 * implementation, we provide support for the latter form, moreover, we
 * allow explicit control of the number of non-zero variables.
 *
 * When no constraints is provided, the full path is generated.
 *
 * Please see the following paper for more details.
 *
 * @code
 * @article{efron2004least,
 *   title={Least angle regression},
 *   author={Efron, B. and Hastie, T. and Johnstone, I. and Tibshirani, R.},
 *   journal={The Annals of statistics},
 *   volume={32},
 *   number={2},
 *   pages={407--499},
 *   year={2004},
 *   publisher={Institute of Mathematical Statistics}
 * }
 * @endcode
 */
class CLeastAngleRegression: public CLinearMachine
{
public:
	/** problem type */
	MACHINE_PROBLEM_TYPE(PT_REGRESSION);

	/** default constructor
	 *
	 * @param lasso - when true, it runs the LASSO, when false, it runs LARS
	 * */
	CLeastAngleRegression(bool lasso=true);

	/** default destructor */
	virtual ~CLeastAngleRegression();

	/** set max number of non-zero variables for early stopping
	 *
	 * @param n 0 means no constraint
	 */
	void set_max_non_zero(int32_t n)
	{
		m_max_nonz = n;
	}

	/** get max number of non-zero variables for early stopping
	 */
	int32_t get_max_non_zero() const
	{
		return m_max_nonz;
	}

	/** set max l1-norm of estimator for early stopping
	 *
	 * @param norm the max l1-norm for beta
	 */
	void set_max_l1_norm(float64_t norm)
	{
		m_max_l1_norm = norm;
	}

	/** get max l1-norm of estimator for early stopping
	 */
	float64_t get_max_l1_norm() const
	{
		return m_max_l1_norm;
	}

	/** switch estimator
	 *
	 * @param num_variable number of non-zero coefficients
	 */
	void switch_w(int32_t num_variable)
	{
		if (w.vlen <= 0)
			SG_ERROR("cannot swith estimator before training")
		if (size_t(num_variable) >= m_beta_idx.size() || num_variable < 0)
			SG_ERROR("cannot switch to an estimator of %d non-zero coefficients", num_variable)
		if (w.vector == NULL)
			w = SGVector<float64_t>(w.vlen);
		std::copy(m_beta_path[m_beta_idx[num_variable]].begin(),
			m_beta_path[m_beta_idx[num_variable]].end(), w.vector);
	}

	/** get path size
	 *
	 * @return the size of variable selection path. Call get_w_for_var(i) to get the
	 * estimator of i-th entry on the path, where i can be in the range [0, path_size)
	 *
	 * @see switch_w
	 * @see get_w_for_var
	 */
	int32_t get_path_size() const
	{
		return m_beta_idx.size();
	}

	/** get w for a particular regularization variable
	 *
	 * @param num_var number of non-zero coefficients
	 *
	 * @return the estimator with num_var non-zero coefficients. **Note** the
	 * returned memory references to some internal structures. The pointer will
	 * become invalid if train is called *again*. So make a copy if you want to
	 * call train multiple times.
	 */
	SGVector<float64_t> get_w_for_var(int32_t num_var)
	{
		return SGVector<float64_t>(&m_beta_path[m_beta_idx[num_var]][0], w.vlen, false);
	}

	/** get classifier type
	 *
	 * @return classifier type LinearRidgeRegression
	 */
	virtual EMachineType get_classifier_type()
	{
		return CT_LARS;
	}

	/** @return object name */
	virtual const char* get_name() const { return "LeastAngleRegression"; }

protected:
	virtual bool train_machine(CFeatures* data=NULL);

private:
	void activate_variable(int32_t v)
	{
		m_num_active++;
		m_active_set.push_back(v);
		m_is_active[v] = true;
	}
	void deactivate_variable(int32_t v_idx)
	{
		m_num_active--;
		m_is_active[m_active_set[v_idx]] = false;
		m_active_set.erase(m_active_set.begin() + v_idx);
	}

	SGMatrix<float64_t> cholesky_insert(SGMatrix<float64_t> X, SGMatrix<float64_t> R, int32_t i_max_corr);
	SGMatrix<float64_t> cholesky_delete(SGMatrix<float64_t> R, int32_t i_kick);


	bool m_lasso; //!< enable lasso modification

	int32_t m_max_nonz;  //!< max number of non-zero variables for early stopping
	float64_t m_max_l1_norm; //!< max l1-norm of beta (estimator) for early stopping

	std::vector<std::vector<float64_t> > m_beta_path;
	std::vector<int32_t> m_beta_idx;

	std::vector<int32_t> m_active_set;
	std::vector<bool> m_is_active;
	int32_t m_num_active;
}; // class LARS

} // namespace shogun

#endif // HAVE_LAPACK
#endif // LEASTANGLEREGRESSION_H__
