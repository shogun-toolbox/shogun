/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Wu Lin
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 * Copyright (C) 2014 Wu Lin
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#ifndef CLAPLACIANINFERENCEMETHODWITHLBFGS_H_
#define CLAPLACIANINFERENCEMETHODWITHLBFGS_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LaplacianInferenceMethod.h>
#include <shogun/mathematics/eigen3.h>


namespace shogun {

/** @brief The Laplace approximation inference method with LBFGS class.
 *
 * This inference method approximates the posterior likelihood function by using
 * Laplace's method. Here, we compute a Gaussian approximation to the posterior
 * via a Taylor expansion around the maximum of the posterior likelihood
 * function.
 *
 * For more details, see "Bayesian Classification with Gaussian Processes" by
 * Christopher K.I Williams and David Barber, published 1998 in the IEEE
 * Transactions on Pattern Analysis and Machine Intelligence, Volume 20, Number
 * 12, Pages 1342-1351.
 *
 * This specific implementation was adapted from the infLaplace.m file in the
 * GPML toolbox.
 */

/** Wrapper class used for the LBFGS minimizer */
  class CSharedInfoForLBFGS {
   public:
    index_t dim;
    Eigen::Map<Eigen::MatrixXd>* kernel;
    Eigen::Map<Eigen::VectorXd>* mean_f;
    CSharedInfoForLBFGS()
        :dim(0), kernel(NULL), mean_f(NULL) {}
  };


  class CLaplacianInferenceMethodWithLBFGS: public CLaplacianInferenceMethod {
   public:
    /** default constructor */
    CLaplacianInferenceMethodWithLBFGS();

    /** constructor
     *
     * @param kernel covariance function
     * @param features features to use in inference
     * @param mean mean function
     * @param labels labels of the features
     * @param model Likelihood model to use
     */
    CLaplacianInferenceMethodWithLBFGS(CKernel* kernel,
                                       CFeatures* features,
                                       CMeanFunction* mean,
                                       CLabels* labels,
                                       CLikelihoodModel* model);

    virtual ~CLaplacianInferenceMethodWithLBFGS();

    /** returns the name of the inference method
     *
     * @return name LaplacianWithLBFGS
     */
    virtual const char* get_name() const {
      return "LaplacianInferenceMethodWithLBFGS";}

   protected:
    /** update alpha using the LBFGS method*/
    virtual void update_alpha();

    /** compute the gradient given the current alpha*/
    virtual void get_gradient_wrt_alpha(Eigen::Map<Eigen::VectorXd>* alpha,
                                        Eigen::Map<Eigen::VectorXd>* gradient);

    /** compute the function value given the current alpha*/
    virtual void get_psi_wrt_alpha(Eigen::Map<Eigen::VectorXd>* alpha,
                                   float64_t* psi);

    /** variables needed to compute gradient and function value*/
    CSharedInfoForLBFGS m_shared;

   private:
    static float64_t evaluate(void *obj,
                              const float64_t *alpha,
                              float64_t *gradient,
                              const int dim,
                              const float64_t step);
  };

} /* namespace shogun */
#endif /* HAVE_EIGEN3 */
#endif /* CLAPLACIANINFERENCEMETHODWITHLBFGS_H_ */
