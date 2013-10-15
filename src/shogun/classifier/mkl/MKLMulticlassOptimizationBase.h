/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Alexander Binder
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef MKLMulticlassOPTIMIZATIONBASE_H_
#define MKLMulticlassOPTIMIZATIONBASE_H_

#include <vector>
#include <shogun/base/SGObject.h>


namespace shogun
{
/** @brief MKLMulticlassOptimizationBase is a helper class for MKLMulticlass.
 *
 *	it is a virtual base class for MKLMulticlassGLPK and MKLMulticlassGradient which are instances of optimization
 *
 */
class MKLMulticlassOptimizationBase: public CSGObject
{
public:
	/** Class default Constructor
	 *
	 */
   MKLMulticlassOptimizationBase();
	/** Class default Destructor
	 *
	 */
   virtual ~MKLMulticlassOptimizationBase();

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
	virtual const char* get_name() const { return "MKLMulticlassOptimizationBase"; }

	/** sets p-norm parameter for MKL
	*	 @param norm the MKL norm
	*/
	virtual void set_mkl_norm(float64_t norm);

protected:


protected:

};
}

#endif
