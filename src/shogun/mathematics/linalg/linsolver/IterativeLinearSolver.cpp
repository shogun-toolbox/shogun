/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/linalg/linsolver/IterativeLinearSolver.h>

namespace shogun
{

template <class T, class ST>
CIterativeLinearSolver<T, ST>::CIterativeLinearSolver()
	: CLinearSolver<T, ST>()
	{
		init();
	}

template <class T, class ST>
CIterativeLinearSolver<T, ST>::CIterativeLinearSolver(bool store_residuals)
	: CLinearSolver<T, ST>()
	{
		init();
		m_store_residuals=store_residuals;
		if (m_store_residuals)
		{
			m_residuals=SGVector<float64_t>(m_max_iteration_limit);
			m_residuals.set_const(0.0);
		}
	}

template <class T, class ST>
void CIterativeLinearSolver<T, ST>::init()
	{
		m_max_iteration_limit=1000;
		m_relative_tolerence=1E-5;
		m_absolute_tolerence=1E-5;
		m_store_residuals=false;
	}

template <class T, class ST>
CIterativeLinearSolver<T, ST>::~CIterativeLinearSolver()
	{
	}

template class CIterativeLinearSolver<float64_t>;
template class CIterativeLinearSolver<complex128_t, float64_t>;
}
