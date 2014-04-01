/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef SPINVCOV_H_
#define SPINVCOV_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** @brief used to estimate inverse covariance matrix using graphical lasso
 *
 * implementation is based on SLEP library's code
 */
class CSparseInverseCovariance : public CSGObject
{
public:

	/** constructor */
	CSparseInverseCovariance();

	/** destructor */
	virtual ~CSparseInverseCovariance();

	/** estimate inverse covariance matrix
	 *
	 * @param S empirical covariance matrix
	 * @param lambda_c regularization constant
	 */
	SGMatrix<float64_t> estimate(SGMatrix<float64_t> S, float64_t lambda_c);

	/** get name */
	const char* get_name() const { return "SparseInverseCovariance"; };


	/** get lasso max iter
	 * @return lasso max iter
	 */
	int32_t get_lasso_max_iter() const { return m_lasso_max_iter; }
	/** get max iter
	 * @return max iter
	 */
	int32_t get_max_iter() const { return m_max_iter; }
	/** get lasso max iter
	 * @return lasso max iter
	 */
	float64_t get_f_gap() const { return m_f_gap; }
	/** get lasso max iter
	 * @return lasso max iter
	 */
	float64_t get_x_gap() const { return m_x_gap; }
	/** get lasso max iter
	 * @return lasso max iter
	 */
	float64_t get_xtol() const { return m_xtol; }

	/** set lasso max iter
	 * @param lasso_max_iter lasso max iter
	 */
	void set_lasso_max_iter(int32_t lasso_max_iter)
	{
		m_lasso_max_iter = lasso_max_iter;
	}
	/** set max iter
	 * @param max_iter max iter
	 */
	void set_max_iter(int32_t max_iter)
	{
		m_max_iter = max_iter;
	}
	/** set f gap
	 * @param f_gap f gap
	 */
	void set_f_gap(int32_t f_gap)
	{
		m_f_gap = f_gap;
	}
	/** set x gap
	 * @param x_gap x gap
	 */
	void set_x_gap(int32_t x_gap)
	{
		m_x_gap = x_gap;
	}
	/** set xtol
	 * @param xtol xtol
	 */
	void set_xtol(int32_t xtol)
	{
		m_xtol = xtol;
	}

private:

	/** register parameters */
	void register_parameters();

protected:

	/** LASSO max iter */
	int32_t m_lasso_max_iter;

	/** max iter */
	int32_t m_max_iter;

	/** fGap */
	float64_t m_f_gap;

	/** xGap */
	float64_t m_x_gap;

	/** xtol */
	float64_t m_xtol;
};

}
#endif
