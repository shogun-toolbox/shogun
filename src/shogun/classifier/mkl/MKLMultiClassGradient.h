/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef MKLMULTICLASSGRADIENT_H_
#define MKLMULTICLASSGRADIENT_H_

#include <vector>
#include <cmath>
#include <cassert>
#include <shogun/base/SGObject.h>
#include <shogun/classifier/mkl/MKLMultiClassOptimizationBase.h>


namespace shogun
{
/** @brief MKLMultiClassGradient is a helper class for MKLMultiClass.
 *
 *	it solves the corresponding linear problem arising in SIP formulation for
 * 	MKL using a gradient based approach
 */
class MKLMultiClassGradient: public MKLMultiClassOptimizationBase
{
public:
	/** Class default Constructor
	 *
	 */
	MKLMultiClassGradient();
	/** Class default Destructor
	 *
	 */
	virtual ~MKLMultiClassGradient();

	/** Class Copy Constructor
	 *
	 */
	MKLMultiClassGradient(MKLMultiClassGradient & gl);

	/** Class Assignment operator
	 *
	 */
	MKLMultiClassGradient operator=(MKLMultiClassGradient & gl);

	/** initializes solver
	 *
	 * @param numkernels2 is the number of kernels
	 *
	 *
	 */
	virtual void setup(const int32_t numkernels2);

	/** adds a constraint to the LP arising in L1 MKL based on two parameters
	 *
	 * @param normw2 is the vector of \f$ \|w_k \|^2 \f$ for all kernels
	 * @param sumofpositivealphas is a term depending on alphas, labels and
	 * biases, see in the function float64_t getsumofsignfreealphas() from
	 * MKLMultiClass.h, it depends on the formulation of the underlying GMNPSVM.
	 *
	 */
	virtual void addconstraint(const ::std::vector<float64_t> & normw2,
			const float64_t sumofpositivealphas);

	/** computes MKL weights
	 *
	 * @param weights2 stores the new weights
	 *
	 */
	virtual void computeweights(std::vector<float64_t> & weights2);

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "MKLMultiClassGradient";
	}

	/** sets p-norm parameter for MKL
	*	 @param norm the MKL norm
	*/
	virtual void set_mkl_norm(float64_t norm);

protected:
	/** helper routine for MKL optimization, computes form manifold coordinates the point on the manifold
	*
	*	@param gammas  - manifold coordinates
	*	@param weights - the point on the manifold
	*
	*/
	void genbetas( ::std::vector<float64_t> & weights ,const ::std::vector<float64_t> & gammas);

	/** helper routine for MKL optimization, computes greadient of manifold parametrization for one coordinate
	*
	*	@param gammagradient  - gradient
	*	@param gammas  - manifold coordinates
	*	@param dim - the coordinate for which thegradient is to be computed
	*
	*/
	void gengammagradient( ::std::vector<float64_t> & gammagradient ,const ::std::vector<float64_t> & gammas,const int32_t dim);
	/** helper routine for MKL optimization, computes optimization objective for one contraint
	*
	*	@param weights - MKL weights
	*	@param index - index of constraint
	*
	*/
	float64_t objectives(const ::std::vector<float64_t> & weights, const int32_t index);

	/** helper routine for MKL optimization, performs linesearch
	*
	* @param finalbeta
	* @param oldweights
	*
	*/
	void linesearch(std::vector<float64_t> & finalbeta,const std::vector<float64_t> & oldweights);

protected:
	/** stores the number of kernels which acts as a parameter for the LP */
	int32_t numkernels;

	/** stores normsofsubkernels which is a constraint, normsofsubkernels[i] belongs to the i-th constraint */
	::std::vector< ::std::vector<float64_t> > normsofsubkernels;
	/** stores the bias type term of constraints, sumsofalphas[i] belongs to the i-th constraint  */
	::std::vector< float64_t > sumsofalphas ;
	/** stores the L^p norm which acts as a parameter for the LP */
	float64_t pnorm;
};
}

#endif
