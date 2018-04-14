/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Saurabh Goyal, Chiyuan Zhang, Bjoern Esser
 */

#ifndef MKLMulticlassGLPK_H_
#define MKLMulticlassGLPK_H_

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/classifier/mkl/MKLMulticlassOptimizationBase.h>

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
