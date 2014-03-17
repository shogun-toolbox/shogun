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
#include <shogun/optimization/lbfgs/lbfgs.h>


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

/*    int             m;*/

    //float64_t epsilon;

    //int             past;

    //float64_t delta;

    //int             max_iterations;

    //int             linesearch;

    //int             max_linesearch;

    //float64_t min_step;

    //float64_t max_step;

    //float64_t ftol;

    //float64_t wolfe;

    //float64_t gtol;

    //float64_t xtol;

    //float64_t orthantwise_c;

    //int             orthantwise_start;

    //int             orthantwise_end;
//} lbfgs_parameter_t;
////
////static const lbfgs_parameter_t _defparam = {
    //6, 1e-5, 0, 1e-5,
    //0, LBFGS_LINESEARCH_DEFAULT, 40,
    //1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
    //0.0, 0, -1,
/*};*/
    virtual lbfgs_parameter_t get_lbfgs_parameter() const;

    virtual void set_lbfgs_parameter(const lbfgs_parameter_t & lbfgs_param,
                                     bool enable_newton_if_fail = true);

    virtual void set_lbfgs_parameter(int m, 
                                     int max_linesearch = 40,
                                     int linesearch = LBFGS_LINESEARCH_DEFAULT,
                                     int max_iterations = 0,
                                     float64_t delta = 1e-5, 
                                     int past = 0, 
                                     float64_t epsilon = 1e-5,
                                     bool enable_newton_if_fail = true,
                                     float64_t min_step = 1e-20,
                                     float64_t max_step = 1e+20,
                                     float64_t ftol = 1e-4,
                                     float64_t wolfe = 0.9,
                                     float64_t gtol = 0.9,
                                     float64_t xtol = 1e-16,
                                     float64_t orthantwise_c = 0.0,
                                     int orthantwise_start = 0,
                                     int orthantwise_end = 1);

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
    CSharedInfoForLBFGS m_shared_variables;

   private:
    // m_variablename where m_ means data member for short
    lbfgs_parameter_t m_lbfgs_param;
    void init();
    static float64_t evaluate(void *obj,
                              const float64_t *alpha,
                              float64_t *gradient,
                              const int dim,
                              const float64_t step);

    bool m_enable_newton_if_fail;
  };

} /* namespace shogun */
#endif /* HAVE_EIGEN3 */
#endif /* CLAPLACIANINFERENCEMETHODWITHLBFGS_H_ */
