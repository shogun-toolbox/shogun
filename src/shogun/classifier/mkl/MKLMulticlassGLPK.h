/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * Update to patch 0.10.0 - thanks to Eric aka Yoo (thereisnoknife@gmail.com)
 *
 */

#ifndef MKLMulticlassGLPK_H_
#define MKLMulticlassGLPK_H_

#include <vector>
#include <base/SGObject.h>
#include <classifier/mkl/MKLMulticlassOptimizationBase.h>

namespace shogun
{
/** @brief MKLMulticlassGLPK is a helper class for MKLMulticlass.
 *
 *	it solves the corresponding linear problem arising in SIP formulation for
 *	MKL using glpk
 */
class MKLMulticlassGLPK: public MKLMulticlassOptimizationBase
{
public:
	/** Class default Constructor
	 *
	 */
   MKLMulticlassGLPK();
	/** Class default Destructor
	 *
	 */
   virtual ~MKLMulticlassGLPK();

	/** initializes GLPK LP sover
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
    * MKLMulticlass.h, it depends on the formulation of the underlying GMNPSVM.
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
	virtual const char* get_name() const { return "MKLMulticlassGLPK"; }

protected:
	/** Class Copy Constructor
	 * protected to avoid its usage because member glp_prob* linearproblem;
	 * from GLPK package is not copyable
	 */
   MKLMulticlassGLPK(MKLMulticlassGLPK & gl);
	/** Class Assignment operator
	 * protected to avoid its usage because member glp_prob* linearproblem;
	 * from GLPK package is not copyable
	 */
   MKLMulticlassGLPK operator=(MKLMulticlassGLPK & gl);

protected:
	/** stores the number of kernels which acts as a parameter for the LP */
	int32_t numkernels;
   /** GLPK data structure of type glp_prob* */
   void* linearproblem;
};
}

#endif
