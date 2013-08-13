/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/JacobiEllipticFunctions.h>
#include <shogun/mathematics/logdet/LinearOperator.h>
#include <shogun/mathematics/logdet/LinearSolver.h>
#include <shogun/mathematics/logdet/EigenSolver.h>
#include <shogun/mathematics/logdet/RationalApproximation.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>

namespace shogun
{

CRationalApproximation::CRationalApproximation()
	: COperatorFunction<float64_t>()
{
	init();

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CRationalApproximation::CRationalApproximation(
	CLinearOperator<float64_t>* linear_operator,
	CIndependentComputationEngine* computation_engine,
	CEigenSolver* eigen_solver,
	index_t num_shifts,
	EOperatorFunction function_type)
	: COperatorFunction<float64_t>(linear_operator, computation_engine,
	  function_type)
{
	init();

	m_eigen_solver=eigen_solver;
	SG_REF(m_eigen_solver);

	m_num_shifts=num_shifts;

	m_shifts=SGVector<complex64_t>(m_num_shifts);
	m_weights=SGVector<complex64_t>(m_num_shifts);

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CRationalApproximation::~CRationalApproximation()
{
	SG_UNREF(m_eigen_solver);

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CRationalApproximation::init()
{
	m_eigen_solver=NULL;
	m_constant_multiplier=0.0;
	m_num_shifts=0;

	SG_ADD((CSGObject**)&m_eigen_solver, "eigen_solver",
		"Eigen solver for computing extremal eigenvalues", MS_NOT_AVAILABLE);

	SG_ADD(&m_shifts, "complex_shifts", "Complex shifts in the linear system",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_weights, "complex_weights", "Complex weights of the linear system",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_constant_multiplier, "constant_multiplier",
		"Constant multiplier in the rational approximation",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_num_shifts, "num_shifts",
		"Number of shifts in the quadrature rule", MS_NOT_AVAILABLE);
}

SGVector<complex64_t> CRationalApproximation::get_shifts() const
{
	return m_shifts;
}

SGVector<complex64_t> CRationalApproximation::get_weights() const
{
	return m_weights;
}

float64_t CRationalApproximation::get_constant_multiplier() const
{
	return m_constant_multiplier;
}

void CRationalApproximation::precompute()
{
	// compute extremal eigenvalues
	m_eigen_solver->compute();
	SG_DEBUG("max_eig=%.15lf\n", m_eigen_solver->get_max_eigenvalue());
	SG_DEBUG("min_eig=%.15lf\n", m_eigen_solver->get_min_eigenvalue());

	compute_shifts_weights_const();
}

void CRationalApproximation::compute_shifts_weights_const()
{
	float64_t PI=M_PI;
	
	// eigenvalues are always real in this case
	float64_t max_eig=m_eigen_solver->get_max_eigenvalue();
	float64_t min_eig=m_eigen_solver->get_min_eigenvalue();

	// we need $(\frac{\lambda_{M}}{\lambda_{m}})^{frac{1}{4}}$ and
	// $(\lambda_{M}*\lambda_{m})^{frac{1}{4}}$ for the rest of the
	// calculation where $lambda_{M}$ and $\lambda_{m} are the maximum
	// and minimum eigenvalue respectively
	float64_t m_div=CMath::pow(max_eig/min_eig, 0.25);
	float64_t m_mult=CMath::pow(max_eig*min_eig, 0.25);
	
	// k=$\frac{(\frac{\lambda_{M}}{\lambda_{m}})^\frac{1}{4}-1}
	// {(\frac{\lambda_{M}}{\lambda_{m}})^\frac{1}{4}+1}$
	float64_t k=(m_div-1)/(m_div+1);
	float64_t L=-CMath::log(k)/PI;
	
	// compute K and K'
	float64_t K=0.0, Kp=0.0;
	CJacobiEllipticFunctions::ellipKKp(L, K, Kp);
	
	// compute constant multiplier
	m_constant_multiplier=-8*K*m_mult/(k*PI*m_num_shifts);
	
	// compute Jacobi elliptic functions sn, cn, dn and compute shifts, weights
	// using conformal mapping of the quadrature rule for discretization of the
	// contour integral
	float64_t m=CMath::sq(k);

	for (index_t i=0; i<m_num_shifts; ++i)
	{
		complex64_t t=complex64_t(0.0, 0.5*Kp)-K+(0.5+i)*2*K/m_num_shifts;

		complex64_t sn, cn, dn;
		CJacobiEllipticFunctions::ellipJC(t, m, sn, cn, dn);

		complex64_t w=m_mult*(1.0+k*sn)/(1.0-k*sn);
		complex64_t dzdt=cn*dn/CMath::sq(1.0/k-sn);

		// compute shifts and weights as per Hale et. al. (2008) [2]
		m_shifts[i]=CMath::sq(w);

		switch (m_function_type)
		{
		case OF_SQRT:
			m_weights[i]=dzdt;
			break;
		case OF_LOG:
			m_weights[i]=2.0*CMath::log(w)*dzdt/w;
			break;
		case OF_POLY:
			SG_NOTIMPLEMENTED
			m_weights[i]=complex64_t(0.0);
			break;
		case OF_UNDEFINED:
			SG_WARNING("Operator function is undefined!\n")
			m_weights[i]=complex64_t(0.0);
			break;
		}
	}
}

}
