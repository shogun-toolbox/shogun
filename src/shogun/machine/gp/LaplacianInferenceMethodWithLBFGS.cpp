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
#include <shogun/optimization/lbfgs/lbfgs.h>

namespace shogun {

CLaplacianInferenceMethodWithLBFGS::CLaplacianInferenceMethodWithLBFGS()
    : CLaplacianInferenceMethod() {
}

CLaplacianInferenceMethodWithLBFGS::CLaplacianInferenceMethodWithLBFGS(
    CKernel* kern,
    CFeatures* feat,
    CMeanFunction* m,
    CLabels* lab,
    CLikelihoodModel* mod)
    : CLaplacianInferenceMethod(kern, feat, m, lab, mod) {
}

CLaplacianInferenceMethodWithLBFGS::~CLaplacianInferenceMethodWithLBFGS() {
}



void CLaplacianInferenceMethodWithLBFGS::get_psi_wrt_alpha(
    Eigen::Map<Eigen::VectorXd>* alpha,
    float64_t* psi) {
  SGVector<float64_t> f(m_shared.dim);
  Eigen::Map<Eigen::VectorXd> eigen_f(f.vector, f.vlen);

  //  f=K*alpha+mean_f given alpha
  eigen_f = (*m_shared.kernel) * ((*alpha) * CMath::sq(m_scale)) \
            + (*m_shared.mean_f);

  //  psi=0.5*alpha.*(f-m)-sum(dlp)
  *psi = alpha->dot(eigen_f - *m_shared.mean_f) * 0.5 -
      SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, f));
}

void CLaplacianInferenceMethodWithLBFGS::get_gradient_wrt_alpha(
    Eigen::Map<Eigen::VectorXd>* alpha,
    Eigen::Map<Eigen::VectorXd>* gradient) {
  SGVector<float64_t> f((m_shared.dim));
  Eigen::Map<Eigen::VectorXd> eigen_f(f.vector, f.vlen);

  //  f=K*alpha+(m_shared.mean_f) given alpha
  eigen_f = (*m_shared.kernel) * ((*alpha) * CMath::sq(m_scale)) \
            + *(m_shared.mean_f);

  SGVector<float64_t> dlp_f \
      = m_model->get_log_probability_derivative_f(m_labels, f, 1);
  Eigen::Map<Eigen::VectorXd> eigen_dlp_f(dlp_f.vector, dlp_f.vlen);

  //  g_alpha=K*(alpha-dlp_f)
  *gradient = *(m_shared.kernel) \
              * ((*alpha - eigen_dlp_f) * CMath::sq(m_scale));
}

}  /* namespace shogun */

#endif /* HAVE_EIGEN3 */
