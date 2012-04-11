/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef LARS_H_WSBLKNVU
#define LARS_H_WSBLKNVU

#include <vector>

#include <shogun/machine/LinearMachine.h>

class CFeatures;

namespace shogun 
{

/** @brief Class for Least Angle Regression, can be used to solve LASSO.
 *
 * LASSO is basically L1 regulairzed least square regression
 * 
 * \f[
 * \|X^T\beta - y\|^2 + \lambda\|\beta\|_{1}
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
class CLARS: public CLinearMachine
{
public:
	CLARS(bool lasso=true):m_lasso(lasso),m_max_nonz(0),m_max_l1_norm(0)
	{
	}
	virtual ~CLARS()
	{
	}

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

	/** load regression from file
	 *
	 * @param srcfile file to load from
	 * @return if loading was successful
	 */
	virtual bool load(FILE* srcfile);

	/** save regression to file
	 *
	 * @param dstfile file to save to
	 * @return if saving was successful
	 */
	virtual bool save(FILE* dstfile);

	/** get classifier type
	 *
	 * @return classifier type LinearRidgeRegression
	 */
	inline virtual EClassifierType get_classifier_type()
	{
		return CT_LARS;
	}

	/** @return object name */
	inline virtual const char* get_name() const { return "LARS"; }

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

	bool m_lasso; //!< enable lasso modification

	int32_t m_max_nonz;  //!< max number of non-zero variables for early stopping
	float64_t m_max_l1_norm; //!< max l1-norm of beta (estimator) for early stopping

	std::vector<std::vector<float64_t> > m_beta_path;

	std::vector<int32_t> m_active_set;
	std::vector<bool> m_is_active;
	int32_t m_num_active;
}; // class LARS

} // namespace shogun

#endif /* end of include guard: LARS_H_WSBLKNVU */

