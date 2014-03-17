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
 * This code specifically adapted from infLaplace.m
 */

#include <shogun/machine/gp/LaplacianInferenceMethodWithLBFGS.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/Math.h>
#include <cstring>

namespace shogun {

CLaplacianInferenceMethodWithLBFGS::CLaplacianInferenceMethodWithLBFGS()
    : CLaplacianInferenceMethod() {
      init();
}

CLaplacianInferenceMethodWithLBFGS::CLaplacianInferenceMethodWithLBFGS(
    CKernel* kern,
    CFeatures* feat,
    CMeanFunction* m,
    CLabels* lab,
    CLikelihoodModel* mod)
    : CLaplacianInferenceMethod(kern, feat, m, lab, mod) {
      init();
}

lbfgs_parameter_t CLaplacianInferenceMethodWithLBFGS::get_lbfgs_parameter() const {
  return m_lbfgs_param;
}
void CLaplacianInferenceMethodWithLBFGS::set_lbfgs_parameter(
    const lbfgs_parameter_t & lbfgs_param,
    bool enable_newton_if_fail) {
  const lbfgs_parameter_t * param_ptr = &lbfgs_param;
  memcpy(&m_lbfgs_param, param_ptr, sizeof(lbfgs_param));
  m_enable_newton_if_fail = enable_newton_if_fail;
}

void CLaplacianInferenceMethodWithLBFGS::set_lbfgs_parameter(int m, 
                                                             int max_linesearch,
                                                             int linesearch,
                                                             int max_iterations,
                                                             float64_t delta, 
                                                             int past, 
                                                             float64_t epsilon,
                                                             bool enable_newton_if_fail,
                                                             float64_t min_step,
                                                             float64_t max_step,
                                                             float64_t ftol,
                                                             float64_t wolfe,
                                                             float64_t gtol,
                                                             float64_t xtol,
                                                             float64_t orthantwise_c,
                                                             int orthantwise_start,
                                                             int orthantwise_end) {
  m_lbfgs_param.m = m;
  m_lbfgs_param.max_linesearch = max_linesearch;
  m_lbfgs_param.linesearch = linesearch;
  m_lbfgs_param.max_iterations = max_iterations;
  m_lbfgs_param.delta = delta;
  m_lbfgs_param.past = past;
  m_lbfgs_param.epsilon = epsilon;
  m_lbfgs_param.min_step = min_step;
  m_lbfgs_param.max_step = max_step;
  m_lbfgs_param.ftol = ftol;
  m_lbfgs_param.wolfe = wolfe;
  m_lbfgs_param.gtol = gtol;
  m_lbfgs_param.xtol = xtol;
  m_lbfgs_param.orthantwise_c = orthantwise_c;
  m_lbfgs_param.orthantwise_start = orthantwise_start;
  m_lbfgs_param.orthantwise_end = orthantwise_end;
  m_enable_newton_if_fail = enable_newton_if_fail;
}

void CLaplacianInferenceMethodWithLBFGS::init(){
  /* Initialize the default parameters for the L-BFGS optimization. */
  lbfgs_parameter_init(&m_lbfgs_param);
  m_enable_newton_if_fail = true;
}

CLaplacianInferenceMethodWithLBFGS::~CLaplacianInferenceMethodWithLBFGS() {
}

float64_t CLaplacianInferenceMethodWithLBFGS::evaluate(void *obj,
                                                       const float64_t *alpha,
                                                       float64_t *gradient,
                                                       const int dim,
                                                       const float64_t step) {
  // Note that alpha = alpha_pre_iter - step * gradient_pre_iter
  /* Unfortunately we can not use dynamic_cast to cast the void * pointer to an
   * object pointer. Therefore, make sure this method is private.  
   */
  CLaplacianInferenceMethodWithLBFGS * obj_prt
      = static_cast<CLaplacianInferenceMethodWithLBFGS *>(obj);
  float64_t * alpha_cast = const_cast<float64_t *>(alpha);
  Eigen::Map<Eigen::VectorXd> eigen_alpha(alpha_cast, dim);
  float64_t psi = 0.0;
  obj_prt->get_psi_wrt_alpha(&eigen_alpha, &psi);
  Eigen::Map<Eigen::VectorXd> eigen_gradient(gradient, dim);
  obj_prt->get_gradient_wrt_alpha(&eigen_alpha, &eigen_gradient);
  return psi;
}

void CLaplacianInferenceMethodWithLBFGS::update_alpha() {
  float64_t psi_new;
  float64_t psi_def;

  // get mean vector and create eigen representation of it
  SGVector<float64_t> mean_f = m_mean->get_mean_vector(m_features);
  Eigen::Map<Eigen::VectorXd> eigen_mean_f(mean_f.vector, mean_f.vlen);

  // create eigen representation of kernel matrix
  Eigen::Map<Eigen::MatrixXd> eigen_ktrtr(m_ktrtr.matrix,
                                          m_ktrtr.num_rows,
                                          m_ktrtr.num_cols);

  // create shogun and eigen representation of function vector
  m_mu = SGVector<float64_t>(mean_f.vlen);  // f
  Eigen::Map<Eigen::VectorXd> eigen_mu(m_mu, m_mu.vlen);

  if (m_alpha.vlen != m_labels->get_num_labels()) {
    // set alpha a zero vector
    m_alpha = SGVector<float64_t>(m_labels->get_num_labels());
    m_alpha.zero();

    // f = mean, if length of alpha and length of y doesn't match
    eigen_mu = eigen_mean_f;
    psi_new = -SGVector<float64_t>::sum(m_model->get_log_probability_f(
            m_labels, m_mu));
  } else {
    // compute f = K * alpha + m
    Eigen::Map<Eigen::VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
    eigen_mu = eigen_ktrtr * (eigen_alpha * CMath::sq(m_scale)) + eigen_mean_f;

    psi_new = eigen_alpha.dot(eigen_mu - eigen_mean_f) / 2.0 -\
        SGVector<float64_t>::sum(
            m_model->get_log_probability_f(m_labels, m_mu));  // f

    psi_def = -SGVector<float64_t>::sum(
        m_model->get_log_probability_f(m_labels, mean_f));  // mean_f

    // if default is better, then use it
    if (psi_def < psi_new) {
      m_alpha.zero();
      eigen_mu = eigen_mean_f;  // f=mean_f
      psi_new = psi_def;
    }
  }
  Eigen::Map<Eigen::VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
  index_t dim = m_alpha.vlen;

  // use for passing variables to compute function value and gradient
  m_shared_variables.dim = dim;
  m_shared_variables.kernel = &eigen_ktrtr;
  m_shared_variables.mean_f = &eigen_mean_f;

  /* In order to use the provided lbfgs function, we have to pass the object via
   * void * pointer, which the evaluate method will use static_cast to cast
   * the pointer to an object pointer.
   * Therefore, make sure the evaluate method is a private method of the class. 
   * Because the evaluate method is defined in a class, we have to pass the
   * method pointer to the lbfgs function via static method
   * If we also use the progress method, make sure the method is static and
   * private. 
   */
  void * obj_prt = static_cast<void *>(this);

  int ret = lbfgs(m_alpha.vlen, m_alpha.vector, &psi_new,
                  CLaplacianInferenceMethodWithLBFGS::evaluate,
                  NULL, obj_prt, &m_lbfgs_param);
  // clean up
  m_shared_variables.dim = 0;
  m_shared_variables.kernel = NULL;
  m_shared_variables.mean_f = NULL;

  /* Note that ret should be zero if the minimization 
   * process terminates without an error.
   * A non-zero value indicates an error. 
   */
  if (m_enable_newton_if_fail && ret != 0 && ret != LBFGS_ALREADY_MINIMIZED) {
   //If some error happened during the L-BFGS optimization, we use the original
   //Newton method.
    SG_WARNING("Error during L-BFGS optimization, using original Newton method as fallback\n");
    CLaplacianInferenceMethod::update_alpha();
    return;
  }

  // compute f = K * alpha + m
  eigen_mu = eigen_ktrtr * (eigen_alpha * CMath::sq(m_scale)) + eigen_mean_f;

  // get log probability derivatives
  dlp  = m_model->get_log_probability_derivative_f(m_labels, m_mu, 1);
  d2lp = m_model->get_log_probability_derivative_f(m_labels, m_mu, 2);
  d3lp = m_model->get_log_probability_derivative_f(m_labels, m_mu, 3);

  // W = -d2lp
  W = d2lp.clone();
  W.scale(-1.0);

  // compute sW
  Eigen::Map<Eigen::VectorXd> eigen_W(W.vector, W.vlen);
  // create shogun and eigen representation of sW
  sW = SGVector<float64_t>(W.vlen);
  Eigen::Map<Eigen::VectorXd> eigen_sW(sW.vector, sW.vlen);

  if (eigen_W.minCoeff() > 0)
    eigen_sW = eigen_W.cwiseSqrt();
  else
    eigen_sW.setZero();
}

void CLaplacianInferenceMethodWithLBFGS::get_psi_wrt_alpha(
    Eigen::Map<Eigen::VectorXd>* alpha,
    float64_t* psi) {
  SGVector<float64_t> f(m_shared_variables.dim);
  Eigen::Map<Eigen::VectorXd> eigen_f(f.vector, f.vlen);

  //  f=K*alpha+mean_f given alpha
  eigen_f = (*m_shared_variables.kernel) * ((*alpha) * CMath::sq(m_scale)) \
            + (*m_shared_variables.mean_f);

  //  psi=0.5*alpha.*(f-m)-sum(dlp)
  *psi = alpha->dot(eigen_f - *m_shared_variables.mean_f) * 0.5 -
      SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, f));
}

void CLaplacianInferenceMethodWithLBFGS::get_gradient_wrt_alpha(
    Eigen::Map<Eigen::VectorXd>* alpha,
    Eigen::Map<Eigen::VectorXd>* gradient) {
  SGVector<float64_t> f((m_shared_variables.dim));
  Eigen::Map<Eigen::VectorXd> eigen_f(f.vector, f.vlen);

  //  f=K*alpha+(m_shared_variables.mean_f) given alpha
  eigen_f = (*m_shared_variables.kernel) * ((*alpha) * CMath::sq(m_scale)) \
            + *(m_shared_variables.mean_f);

  SGVector<float64_t> dlp_f \
      = m_model->get_log_probability_derivative_f(m_labels, f, 1);
  Eigen::Map<Eigen::VectorXd> eigen_dlp_f(dlp_f.vector, dlp_f.vlen);

  //  g_alpha=K*(alpha-dlp_f)
  *gradient = *(m_shared_variables.kernel) \
              * ((*alpha - eigen_dlp_f) * CMath::sq(m_scale));
}

}  /* namespace shogun */

#endif /* HAVE_EIGEN3 */
