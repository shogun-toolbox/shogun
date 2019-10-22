/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Sunil Mahendrakar, Bjoern Esser
 */

#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/mathematics/JacobiEllipticFunctions.h>
#endif //USE_GPL_SHOGUN
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
#include <shogun/mathematics/linalg/eigsolver/EigenSolver.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>

namespace shogun
{

RationalApproximation::RationalApproximation()
	: OperatorFunction<float64_t>()
{
	init();

	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

RationalApproximation::RationalApproximation(
	std::shared_ptr<LinearOperator<float64_t>> linear_operator, std::shared_ptr<EigenSolver> eigen_solver,
	float64_t desired_accuracy, EOperatorFunction function_type)
	: OperatorFunction<float64_t>(linear_operator, function_type)
{
	init();

	m_eigen_solver=eigen_solver;
	

	m_desired_accuracy=desired_accuracy;

	SG_TRACE("{} created ({})", this->get_name(), fmt::ptr(this));
}

RationalApproximation::~RationalApproximation()
{
	

	SG_TRACE("{} destroyed ({})", this->get_name(), fmt::ptr(this));
}

void RationalApproximation::init()
{
	m_eigen_solver=NULL;
	m_constant_multiplier=0.0;
	m_num_shifts=0;
	m_desired_accuracy=0.0;

	SG_ADD((std::shared_ptr<SGObject>*)&m_eigen_solver, "eigen_solver",
		"Eigen solver for computing extremal eigenvalues");

	SG_ADD(&m_shifts, "complex_shifts", "Complex shifts in the linear system");

	SG_ADD(&m_weights, "complex_weights", "Complex weights of the linear system");

	SG_ADD(&m_constant_multiplier, "constant_multiplier",
		"Constant multiplier in the rational approximation");

	SG_ADD(&m_num_shifts, "num_shifts",
		"Number of shifts in the quadrature rule");

	SG_ADD(&m_desired_accuracy, "desired_accuracy",
		"Desired accuracy of the rational approximation");
}

SGVector<complex128_t> RationalApproximation::get_shifts() const
{
	return m_shifts;
}

SGVector<complex128_t> RationalApproximation::get_weights() const
{
	return m_weights;
}

float64_t RationalApproximation::get_constant_multiplier() const
{
	return m_constant_multiplier;
}

index_t RationalApproximation::get_num_shifts() const
{
	return m_num_shifts;
}

void RationalApproximation::set_num_shifts(index_t num_shifts)
{
	m_num_shifts=num_shifts;
}

void RationalApproximation::precompute()
{
	// compute extremal eigenvalues
	m_eigen_solver->compute();
	io::info("max_eig={:.15f}", m_eigen_solver->get_max_eigenvalue());
	io::info("min_eig={:.15f}", m_eigen_solver->get_min_eigenvalue());

	require(m_eigen_solver->get_min_eigenvalue()>0,
		"Minimum eigenvalue is negative, please provide a Hermitian matrix");

	// compute number of shifts from accuracy if shifts are not set yet
	if (m_num_shifts==0)
		m_num_shifts=compute_num_shifts_from_accuracy();

	io::info("Computing {} shifts", m_num_shifts);
	compute_shifts_weights_const();
}

int32_t RationalApproximation::compute_num_shifts_from_accuracy()
{
	require(m_desired_accuracy>0, "Desired accuracy must be positive but is {}",
			m_desired_accuracy);

	float64_t max_eig=m_eigen_solver->get_max_eigenvalue();
	float64_t min_eig=m_eigen_solver->get_min_eigenvalue();

	float64_t log_cond_number = std::log(max_eig) - std::log(min_eig);
	float64_t two_pi_sq=2.0*M_PI*M_PI;

	int32_t num_shifts = static_cast<index_t>(
		-1.5 * (log_cond_number + 6.0) * std::log(m_desired_accuracy) /
		two_pi_sq);

	return num_shifts;
}

void RationalApproximation::compute_shifts_weights_const()
{
	float64_t PI=M_PI;

	// eigenvalues are always real in this case
	float64_t max_eig=m_eigen_solver->get_max_eigenvalue();
	float64_t min_eig=m_eigen_solver->get_min_eigenvalue();

	// we need $(\frac{\lambda_{M}}{\lambda_{m}})^{frac{1}{4}}$ and
	// $(\lambda_{M}*\lambda_{m})^{frac{1}{4}}$ for the rest of the
	// calculation where $lambda_{M}$ and $\lambda_{m} are the maximum
	// and minimum eigenvalue respectively
	float64_t m_div=Math::pow(max_eig/min_eig, 0.25);
	float64_t m_mult=Math::pow(max_eig*min_eig, 0.25);

	// k=$\frac{(\frac{\lambda_{M}}{\lambda_{m}})^\frac{1}{4}-1}
	// {(\frac{\lambda_{M}}{\lambda_{m}})^\frac{1}{4}+1}$
	float64_t k=(m_div-1)/(m_div+1);
	float64_t L = -std::log(k) / PI;

	// compute K and K'
	float64_t K=0.0, Kp=0.0;
#ifdef USE_GPL_SHOGUN
	JacobiEllipticFunctions::ellipKKp(L, K, Kp);
#else
	gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN

	// compute constant multiplier
	m_constant_multiplier=-8*K*m_mult/(k*PI*m_num_shifts);

	// compute Jacobi elliptic functions sn, cn, dn and compute shifts, weights
	// using conformal mapping of the quadrature rule for discretization of the
	// contour integral
	float64_t m=Math::sq(k);

	// allocate memory for shifts
	m_shifts=SGVector<complex128_t>(m_num_shifts);
	m_weights=SGVector<complex128_t>(m_num_shifts);

	for (index_t i=0; i<m_num_shifts; ++i)
	{
		complex128_t t=complex128_t(0.0, 0.5*Kp)-K+(0.5+i)*2*K/m_num_shifts;

		complex128_t sn, cn, dn;
#ifdef USE_GPL_SHOGUN
		JacobiEllipticFunctions::ellipJC(t, m, sn, cn, dn);
#else
		gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN

		complex128_t w=m_mult*(1.0+k*sn)/(1.0-k*sn);
		complex128_t dzdt=cn*dn/Math::sq(1.0/k-sn);

		// compute shifts and weights as per Hale et. al. (2008) [2]
		m_shifts[i]=Math::sq(w);

		switch (m_function_type)
		{
		case OF_SQRT:
			m_weights[i]=dzdt;
			break;
		case OF_LOG:
			m_weights[i] = 2.0 * std::log(w) * dzdt / w;
			break;
		case OF_POLY:
			not_implemented(SOURCE_LOCATION);
			m_weights[i]=complex128_t(0.0);
			break;
		case OF_UNDEFINED:
			io::warn("Operator function is undefined!");
			m_weights[i]=complex128_t(0.0);
			break;
		}
	}
}

}
