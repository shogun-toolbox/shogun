/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef RATIONAL_APPROXIMATION_H_
#define RATIONAL_APPROXIMATION_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/logdet/OperatorFunction.h>

namespace shogun
{

template<class T> class SGVector;
template<class T> class CLinearOperator;
class CIndependentComputationEngine;
class CJobResultAggregator;
class CEigenSolver;

/** @brief Abstract base class of the rational approximation of a function of a
 * linear operator (A) times vector (v) using Cauchy's integral formula -
 * \f[f(\text{A})\text{v}=\oint_{\Gamma}f(z)(z\text{I}-\text{A})^{-1}
 * \text{v}dz\f]
 * Computes eigenvalues of linear operator and uses Jacobi elliptic functions 
 * and conformal maps [2] for quadrature rule for discretizing the contour
 * integral and computes complex shifts, weights and constant multiplier of the
 * rational approximation of the above expression as
 * \f[f(\text{A})\text{v}\approx \eta\text{A}\Im-\left(\sum_{l=1}^{N}\alpha_{l}
 * (\text{A}-\sigma_{l}\text{I})^{-1}\text{v}\right)\f]
 * where \f$\alpha_{l},\sigma_{l}\in\mathbb{C}\f$ are respectively the shifts
 * and weights of the linear systems generated from the rational approximation,
 * and \f$\eta\in\mathbb{R}\f$ is the constant multiplier, equals to
 * \f$\frac{-8K(\lambda_{m}\lambda_{M})^{\frac{1}{4}}}{k\pi N}\f$.
 *
 * Reference:
 * [1] Aune, E., D. Simpson, and J. Eidsvik (2012). Parameter estimation
 * in high dimensional gaussian distributions. Technical Report Statistics
 * 5/2012, NTNU.
 *
 * [2] Nicholas Hale, Nicholas J. Higham and Lloyd N. Trefethen (2008).
 * Computing \f$A^{\alpha}\f$ , \f$log(A)\f$ and related matrix functions by
 * contour integrals. SIAM Journal of Numerical Analysis, 46:2505-2523
 *
 * Note: The implementation of compute_weights_shifts_const function has been
 * adapted from KRYLSTAT (Copyright 2011 by Erlend Aune <erlenda@math.ntnu.no>)
 * under GPL2+. See https://github.com/Froskekongen/KRYLSTAT.
 */
class CRationalApproximation : public COperatorFunction<float64_t>
{
public:
	/** default constructor */
	CRationalApproximation();

	/** 
	 * constructor
	 *
	 * @param linear_operator real valued linear operator for this operator
	 * function
	 * @param computation_engine engine that computes the independent jobs
	 * @param eigen_solver eigen solver for computing min and max eigenvalues
	 * needed for computing shifts, weights and constant multiplier
	 * @param num_shifts number of contour points in the quadrature rule of
	 * of discretization of the contour integral
	 * @param function_type operator function type
	 */
	CRationalApproximation(
		CLinearOperator<float64_t>* linear_operator,
		CIndependentComputationEngine* computation_engine,
		CEigenSolver* eigen_solver,
		index_t num_shifts,
		EOperatorFunction function_type);

	/** destructor */
	virtual ~CRationalApproximation();

	/** 
	 * precompute method that computes extremal eigenvalues using the eigensolver
	 * and then computes complex shifts, weights and constant multiplier coming
	 * from rational approximation of operator function times vector
	 */
	virtual void precompute();

	/** 
	 * abstract method that creates a job result aggregator, then creates a
	 * number of jobs based on its implementation, attaches the aggregator
	 * with all those jobs, hands over the responsility of those to the
	 * computation engine and then returns the aggregator for collecting the
	 * job results
	 * 
	 * @param sample the vector for which new computation job(s) are to be created
	 * @return the array of generated independent jobs
	 */
	virtual CJobResultAggregator* submit_jobs(SGVector<float64_t> sample) = 0;

	/** @return shifts */
	SGVector<complex64_t> get_shifts() const;

	/** @return weights */
	SGVector<complex64_t> get_weights() const;

	/** @return constant multiplier */
	float64_t get_constant_multiplier() const;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CRationalApproximation";
	}

protected:
	/** the eigen solver for computing extremal eigenvalues */
	CEigenSolver* m_eigen_solver;

	/** complex shifts in the systems coming from rational approximation */
	SGVector<complex64_t> m_shifts;

	/** complex weights in the systems coming from rational approximation */
	SGVector<complex64_t> m_weights;

	/** constant multiplier  */
	float64_t m_constant_multiplier;

	/** number of shifts */
	index_t m_num_shifts;

private:
	/** initializes with default values and registers params */
	void init();

	/** 
	 * computes complex integration shifts and weights using conformal mapping
	 * of quadrature rule of discretization of the contour integral
	 */
	void compute_shifts_weights_const();
};

}

#endif // RATIONAL_APPROXIMATION_H_
