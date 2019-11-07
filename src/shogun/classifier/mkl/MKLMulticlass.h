/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Chiyuan Zhang, Viktor Gal, 
 *          Bjoern Esser, Saurabh Goyal
 */

#ifndef MKLMulticlass_H_
#define MKLMulticlass_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/multiclass/GMNPSVM.h>
#include <shogun/classifier/mkl/MKLMulticlassGLPK.h>
#include <shogun/classifier/mkl/MKLMulticlassGradient.h>
#include <shogun/multiclass/MulticlassSVM.h>


namespace shogun
{

/** @brief MKLMulticlass is a class for L1-norm Multiclass MKL.
 *
 *	It is based on the GMNPSVM Multiclass SVM.
 *	Its own parameters are the L2 norm weight change based MKL
 *
 *	Its termination criterion is set by void set_mkl_epsilon(float64_t eps ); and
 *	the maximal number of MKL iterations is set by void
 *	set_max_num_mkliters(int32_t maxnum); It passes the regularization
 *	constants C1 and C2 to GMNPSVM.
 */
class MKLMulticlass : public MulticlassSVM
{
public:
   /** Class default Constructor */
   MKLMulticlass();

   /** Class Constructor commonly used in Shogun Toolbox
    * @param C constant C
    * @param k kernel
    * @param lab labels
    */
   MKLMulticlass(float64_t C, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab);

   /** Class default Destructor */
   virtual ~MKLMulticlass();

   /** get classifier type
    *
    * @return classifier type GMNPMKL
    */
   virtual inline EMachineType get_classifier_type()
   { return CT_MKLMULTICLASS; }

   /** returns MKL weights for the different kernels
    *
    * @param numweights is output parameter, is set to zero if no weights
    * have been computed or to the number of MKL weights which is equal to the number of kernels
    *
    * @return NULL if no weights have been computed or otherwise an array
    * with the weights, caller has to delete[] the output by itself
    */
   float64_t* getsubkernelweights(int32_t & numweights);

   /** sets MKL termination threshold
    *
    * @param eps is the desired threshold value
    * the termination criterion is the L2 norm between the current MKL weights
    *  and their counterpart from the previous iteration
    *
    */
   void set_mkl_epsilon(float64_t eps );

   /** sets maximal number of MKL iterations
    *
    * @param maxnum is the desired maximal number of MKL iterations; when it
    *  is reached the MKL terminates irrespective of the MKL progress
    * set it to a nonpositive value in order to turn it off
    *
    */
   void set_max_num_mkliters(int32_t maxnum);

   /** set mkl norm
    * @param norm
    */
   virtual void set_mkl_norm(float64_t norm);

protected:
   /** Class Copy Constructor
    * protected to avoid its usage
    *
    */
   MKLMulticlass( const MKLMulticlass & cm);

   /** Class Assignment operator
    * protected to avoid its usage
    *
    */
   MKLMulticlass operator=( const MKLMulticlass & cm);

   /** performs some sanity checks (on the provided kernel), inits the
    * GLPK-based LP solver
    *
    */
   void initlpsolver();

   /** inits the underlying Multiclass SVM
    *
    */
   void initsvm();


   /** checks MKL for convergence
    *
    * @param numberofsilpiterations is the number of currently done iterations
    *
    */
   virtual bool evaluatefinishcriterion(const int32_t
         numberofsilpiterations);


   /** adds a constraint to the LP used in MKL
    *
    * @param curweights are the current MKL weights
    *
    * it uses
    * void addingweightsstep( const std::vector<float64_t> & curweights);
    * and
    * float64_t getsumofsignfreealphas();
    */
   void addingweightsstep( const std::vector<float64_t> & curweights);

   /** computes the first svm-dependent part used for generating MKL constraints
    * it is
    * \f$ \sum_y b_y^2-\sum_i \sum_{ y | y \neq y_i} \alpha_{iy}(b_{y_i}-b_y-1) \f$
    */
   float64_t getsumofsignfreealphas();

   /** computes the second svm-dependent part used for generating MKL
    * constraints
    *
    * @param ind is the index of the kernel for which
    * to compute \f$ \|w \|^2  \f$
    */
   float64_t getsquarenormofprimalcoefficients(
         const int32_t ind);

   /** train Multiclass MKL classifier
    *
    * @param data training data (parameter can be avoided if distance or
    * kernel-based classifiers are used and distance/kernels are
    * initialized with train data)
    *
    * @return whether training was successful
    */
   virtual bool train_machine(std::shared_ptr<Features> data=NULL);

   /** @return object name */
    virtual const char* get_name() const { return "MKLMulticlass"; }

protected:
   /** the Multiclass svm for fixed MKL weights
   *
   *
   */
   std::shared_ptr<GMNPSVM> svm;

   /** the solver wrapper */
   std::shared_ptr<MKLMulticlassOptimizationBase> lpw;

   /** stores the last two mkl iteration weights */
   ::std::vector< std::vector< float64_t> > weightshistory;

   /** MKL termination threshold
   *	is set by void set_mkl_epsilon(float64_t eps );
   */
   float64_t mkl_eps;

   /** maximal number of MKL iterations
   *	is set by void set_max_num_mkliters(int32_t maxnum);
   */
   int32_t max_num_mkl_iters;

   /** MKL norm >=1
   *
   */
   float64_t pnorm;

   /** stores the term
	* \f$\| w_l \|^2 = \alpha Y K_l Y \alpha\f$
   *
   */
   std::vector<float64_t> normweightssquared;

   /** norm term above from previous iteration */
   std::vector<float64_t> oldnormweightssquared;

   /** first svm-dependent part used for generating MKL constraints */
   float64_t curalphaterm;
   /** alpha term above from previous iteration */
   float64_t oldalphaterm;
};

}
#endif // GMNPMKL_H_
