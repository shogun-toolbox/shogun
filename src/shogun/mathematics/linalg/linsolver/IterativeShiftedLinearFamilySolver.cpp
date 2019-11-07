/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/linsolver/IterativeShiftedLinearFamilySolver.h>

namespace shogun
{

template <class T, class ST>
IterativeShiftedLinearFamilySolver<T, ST>::IterativeShiftedLinearFamilySolver()
	: IterativeLinearSolver<T, T>()
	{
	}

template <class T, class ST>
IterativeShiftedLinearFamilySolver<T, ST>
	::IterativeShiftedLinearFamilySolver(bool store_residuals)
	: IterativeLinearSolver<T, T>(store_residuals)
	{
	}

template <class T, class ST>
IterativeShiftedLinearFamilySolver<T, ST>::~IterativeShiftedLinearFamilySolver()
	{
	}

	template <class T, class ST>
	void IterativeShiftedLinearFamilySolver<T, ST>::compute_zeta_sh_new(
	    const SGVector<ST>& zeta_sh_old, const SGVector<ST>& zeta_sh_cur,
	    const SGVector<ST>& shifts, const T& beta_old, const T& beta_cur,
	    const T& alpha, SGVector<ST>& zeta_sh_new, bool negate)
	{
		// compute zeta_sh_new according to Jergerlehner, eq. 2.44
		// [see IterativeShiftedLinearFamilySolver.h]
		for (index_t i=0; i<zeta_sh_new.vlen; ++i)
		{
			ST shift_value = shifts[i];
			if (negate)
				shift_value = -shifts[i];
			ST numer=zeta_sh_old[i]*zeta_sh_cur[i]*beta_old;

			ST denom =
			    beta_cur * alpha * (zeta_sh_old[i] - zeta_sh_cur[i]) +
			    beta_old * zeta_sh_old[i] * (1.0 - beta_cur * shift_value);

			// handle division by zero
			if (denom==static_cast<ST>(0.0))
				denom=static_cast<ST>(Math::MACHINE_EPSILON);

			zeta_sh_new[i]=numer/denom;
	  }
	}

template <class T, class ST>
void IterativeShiftedLinearFamilySolver<T, ST>::compute_beta_sh(
		const SGVector<ST>& zeta_sh_new, const SGVector<ST>& zeta_sh_cur, const T& beta_cur,
		SGVector<ST>& beta_sh_cur)
	{
		// compute beta_sh_cur according to Jergerlehner, eq. 2.42
		// [see IterativeShiftedLinearFamilySolver.h]
		for (index_t i=0; i<beta_sh_cur.vlen; ++i)
		{
			ST numer=beta_cur*zeta_sh_new[i];
			ST denom=zeta_sh_cur[i];

			// handle division by zero
			if (denom==static_cast<ST>(0.0))
				denom=static_cast<ST>(Math::MACHINE_EPSILON);

			beta_sh_cur[i]=numer/denom;
		}
	}

template <class T, class ST>
void IterativeShiftedLinearFamilySolver<T, ST>::compute_alpha_sh(
		const SGVector<ST>& zeta_sh_cur, const SGVector<ST>& zeta_sh_old,
		const SGVector<ST>& beta_sh_old, const T& beta_old, const T& alpha, SGVector<ST>& alpha_sh)
	{
		// compute alpha_sh_cur according to Jergerlehner, eq. 2.43
		// [see IterativeShiftedLinearFamilySolver.h]
	for (index_t i=0; i<alpha_sh.vlen; ++i)
	  {
			ST numer=alpha*zeta_sh_cur[i]*beta_sh_old[i];
			ST denom=zeta_sh_old[i]*beta_old;

			// handle division by zero
			if (denom==static_cast<ST>(0.0))
				denom=static_cast<ST>(Math::MACHINE_EPSILON);

			alpha_sh[i]=numer/denom;
	  }
	}

template class IterativeShiftedLinearFamilySolver<float64_t>;
template class IterativeShiftedLinearFamilySolver<float64_t, complex128_t>;
}
