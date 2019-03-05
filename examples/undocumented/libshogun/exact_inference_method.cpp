/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Eleftherios Avramidis
 */

#include <shogun/base/init.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/Math.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/ConstMean.h>
#include <iostream>

#include <shogun/lib/any.h>

using namespace shogun;

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

    // create some easy regression data: 1d noisy sine wave
    index_t ntr=5;

    SGMatrix<float64_t> feat_train(1, ntr);
    SGVector<float64_t> lab_train(ntr);

    feat_train[0]=1.25107;
    feat_train[1]=2.16097;
    feat_train[2]=0.00034;
    feat_train[3]=0.90699;
    feat_train[4]=0.44026;

    lab_train[0]=0.39635;
    lab_train[1]=0.00358;
    lab_train[2]=-1.18139;
    lab_train[3]=1.35533;
    lab_train[4]=-0.08232;

    // shogun representation of features and labels
    CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
    CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

    float64_t ell=0.1;

    // choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean function
    CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);


    CZeroMean* mean=new CZeroMean();

    // Gaussian likelihood with sigma = 0.25
    CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

    // specify GP regression with exact inference
    CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, features_train,
                                                         mean, labels_train, lik);
    // build parameter dictionary
    CMap<AnyParameter*, CSGObject*>* parameter_dictionary=new CMap<AnyParameter*, CSGObject*>();
    inf->build_gradient_parameter_dictionary(parameter_dictionary);


//    // compute derivatives wrt parameters
//    CMap<TParameter*, SGVector<float64_t> >* gradient=
//            inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

//    // get parameters to compute derivatives
//    TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
//    TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");
//    TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("log_sigma");
//
//    float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
//    float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];
//    float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];

    // comparison of partial derivatives of negative marginal likelihood with
    // result from GPML package:
    // lik =  0.10638
    // cov =
    // -0.015133
    // 1.699483
//    EXPECT_NEAR(dnlZ_lik, 0.10638, 1E-5);
//    EXPECT_NEAR(dnlZ_ell, -0.015133, 1E-6);
//    EXPECT_NEAR(dnlZ_sf2, 1.699483, 1E-6);

//    SG_UNREF(gradient);
    SG_UNREF(parameter_dictionary);
    SG_UNREF(inf);

	exit_shogun();

	return 0;
}

