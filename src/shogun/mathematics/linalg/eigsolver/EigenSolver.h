/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef EIGEN_SOLVER_H_
#define EIGEN_SOLVER_H_

#include <shogun/lib/config.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

/** @brief Abstract base class that provides an abstract compute method for
 * computing eigenvalues of a real valued, self-adjoint linear operator. It
 * also provides method for getting min and max eigenvalues.
 */
class CEigenSolver : public CSGObject
{
public:
	/** default constructor */
	CEigenSolver()
	: CSGObject()
	{
		init();
	}

	/**
	 * constructor
	 *
	 * @param linear_operator real valued self-adjoint linear operator
	 * whose eigenvalues have to be found
	 */
	CEigenSolver(CLinearOperator<float64_t>* linear_operator)
	: CSGObject()
	{
		init();

		m_linear_operator=linear_operator;
		SG_REF(m_linear_operator);
	}
	/** destructor */
	virtual ~CEigenSolver()
	{
		SG_UNREF(m_linear_operator);
	}

	/**
	 * abstract compute method for computing eigenvalues of a real
	 * valued linear operator
	 */
	virtual void compute() = 0;

	/** sets the min eigelvalue of a real valued self-adjoint linear operator */
	void set_min_eigenvalue(float64_t min_eigenvalue)
	{
		m_min_eigenvalue=min_eigenvalue;
		m_is_computed_min=true;
	}

	/** @return min eigenvalue of a real valued self-adjoint linear operator */
	const float64_t get_min_eigenvalue() const
	{
		return m_min_eigenvalue;
	}

	/** sets the max eigelvalue of a real valued self-adjoint linear operator */
	void set_max_eigenvalue(float64_t max_eigenvalue)
	{
		m_max_eigenvalue=max_eigenvalue;
		m_is_computed_max=true;
	}

	/** @return max eigenvalue of a real valued self-adjoint linear operator */
	const float64_t get_max_eigenvalue() const
	{
		return m_max_eigenvalue;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "EigenSolver";
	}

#ifdef HAVE_LAPACK
	/** Compute eigenvalues and eigenvectors of symmetric matrix using
        * LAPACK
        *
        * @param matrix symmetric matrix to compute eigenproblem. Is
        * overwritten and contains orthonormal eigenvectors afterwards
        * @return eigenvalues vector with eigenvalues equal to number of rows
        * in matrix
        * */      
	SGVector<float64_t> compute_eigenvectors(SGMatrix<float64_t> matrix) const
	{
		if (matrix.num_rows!=matrix.num_cols)
		{
			SG_SERROR("SGMatrix::compute_eigenvectors(SGMatrix<float64_t>): matrix"
			" rows and columns are not equal!\n");
		}
		/* use reference counting for SGVector */
		SGVector<float64_t> result(NULL, 0, true);
		result.vlen=matrix.num_rows;
		result.vector=compute_eigenvectors(matrix.matrix, matrix.num_rows,
		matrix.num_rows);
		return result;
	}


	/** compute eigenvalues and eigenvectors of symmetric matrix
	*
	* @param matrix overwritten and contains n orthonormal eigenvectors
	* @param n
	* @param m
	* @return eigenvalues (array of length n, to be deleted[])
	* */
	float64_t* compute_eigenvectors(double* matrix, int n, int m) const
	{
		ASSERT(n == m)

		char V='V';
		char U='U';
		int info;
		int ord=n;
		int lda=n;
		double* eigenvalues=SG_CALLOC(float64_t, n+1);

		// lapack sym matrix eigenvalues+vectors
		wrap_dsyev(V, U,  ord, matrix, lda,
				eigenvalues, &info);

		if (info!=0)
			SG_SERROR("DSYEV failed with code %d\n", info)

		return eigenvalues;
	}
#endif

protected:
	/** min eigenvalue */
	float64_t m_min_eigenvalue;

	/** max eigenvalue */
	float64_t m_max_eigenvalue;

	/** the linear solver whose eigenvalues have to be found */
	CLinearOperator<float64_t>* m_linear_operator;

	/** flag that denotes that the minimum eigenvalue is already computed */
	bool m_is_computed_min;

	/** flag that denotes that the maximum eigenvalue is already computed */
	bool m_is_computed_max;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_min_eigenvalue=0.0;
		m_max_eigenvalue=0.0;
		m_linear_operator=NULL;
		m_is_computed_min=false;
		m_is_computed_max=false;

		SG_ADD(&m_min_eigenvalue, "min_eigenvalue",
			"Minimum eigenvalue of a real valued self-adjoint linear operator",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_max_eigenvalue, "max_eigenvalue",
			"Maximum eigenvalue of a real valued self-adjoint linear operator",
			MS_NOT_AVAILABLE);

		SG_ADD((CSGObject**)&m_linear_operator, "linear_operator",
			"Self-adjoint linear operator",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_is_computed_min, "is_computed_min",
			"Flag denoting that the minimum eigenvalue has already been computed",
			MS_NOT_AVAILABLE);

		SG_ADD(&m_max_eigenvalue, "is_computed_max",
			"Flag denoting that the maximum eigenvalue has already been computed",
			MS_NOT_AVAILABLE);

	}
};

}

#endif // EIGEN_SOLVER_H_
