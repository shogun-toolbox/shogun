/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* Remove C Prefix */
%shared_ptr(shogun::MeanFunction)
%shared_ptr(shogun::ZeroMean)
%shared_ptr(shogun::ConstMean)

%shared_ptr(shogun::Inference)
%shared_ptr(shogun::Seedable<shogun::Inference>)
%shared_ptr(shogun::RandomMixin<shogun::Inference, std::mt19937_64>)
%shared_ptr(shogun::ExactInferenceMethod)
%shared_ptr(shogun::LaplaceInference)
%shared_ptr(shogun::SparseInference)
%shared_ptr(shogun::SingleSparseInference)
%shared_ptr(shogun::SingleFITCInference)
%shared_ptr(shogun::SingleLaplaceInferenceMethod)
%shared_ptr(shogun::MultiLaplaceInferenceMethod)
%shared_ptr(shogun::FITCInferenceMethod)
%shared_ptr(shogun::SingleFITCLaplaceInferenceMethod)
%shared_ptr(shogun::VarDTCInferenceMethod)
%shared_ptr(shogun::EPInferenceMethod)

%shared_ptr(shogun::LikelihoodModel)
%shared_ptr(shogun::Seedable<shogun::LikelihoodModel>)
%shared_ptr(shogun::RandomMixin<shogun::LikelihoodModel, std::mt19937_64>)
%shared_ptr(shogun::ProbitLikelihood)
%shared_ptr(shogun::LogitLikelihood)
%shared_ptr(shogun::SoftMaxLikelihood)
%shared_ptr(shogun::GaussianLikelihood)
%shared_ptr(shogun::StudentsTLikelihood)

%shared_ptr(shogun::VariationalLikelihood)
%shared_ptr(shogun::VariationalGaussianLikelihood)
%shared_ptr(shogun::NumericalVGLikelihood)
%shared_ptr(shogun::DualVariationalGaussianLikelihood)
%shared_ptr(shogun::LogitVGLikelihood)
%shared_ptr(shogun::LogitVGPiecewiseBoundLikelihood)
%shared_ptr(shogun::LogitDVGLikelihood)
%shared_ptr(shogun::ProbitVGLikelihood)
%shared_ptr(shogun::StudentsTVGLikelihood)

%shared_ptr(shogun::KLInference)
%shared_ptr(shogun::KLLowerTriangularInference)
%shared_ptr(shogun::KLCovarianceInferenceMethod)
%shared_ptr(shogun::KLDiagonalInferenceMethod)
%shared_ptr(shogun::KLCholeskyInferenceMethod)
%shared_ptr(shogun::KLDualInferenceMethod)

%shared_ptr(shogun::KLDualInferenceMethodMinimizer)

%shared_ptr(shogun::GaussianProcessMachine)
%shared_ptr(shogun::GaussianProcessClassification)
%shared_ptr(shogun::GaussianProcessRegression)


/* These functions return new Objects */

/* Include Class Headers to make them visible from within the target language */
%include <shogun/evaluation/DifferentiableFunction.h>
%include <shogun/machine/gp/LikelihoodModel.h>
%template(SeeableLikelihoodModel) shogun::Seedable<shogun::LikelihoodModel>;
%template(RandomMixinLikelihoodModel) shogun::RandomMixin<shogun::LikelihoodModel, std::mt19937_64>;
%include <shogun/machine/gp/ProbitLikelihood.h>
%include <shogun/machine/gp/LogitLikelihood.h>
%include <shogun/machine/gp/SoftMaxLikelihood.h>
%include <shogun/machine/gp/GaussianLikelihood.h>
%include <shogun/machine/gp/StudentsTLikelihood.h>

%include <shogun/machine/gp/VariationalLikelihood.h>
%include <shogun/machine/gp/VariationalGaussianLikelihood.h>
%include <shogun/machine/gp/NumericalVGLikelihood.h>
%include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>
%include <shogun/machine/gp/LogitVGLikelihood.h>
%include <shogun/machine/gp/LogitVGPiecewiseBoundLikelihood.h>
%include <shogun/machine/gp/LogitDVGLikelihood.h>
%include <shogun/machine/gp/ProbitVGLikelihood.h>
%include <shogun/machine/gp/StudentsTVGLikelihood.h>

%include <shogun/machine/gp/MeanFunction.h>
%include <shogun/machine/gp/ZeroMean.h>
%include <shogun/machine/gp/ConstMean.h>

%include <shogun/machine/gp/Inference.h>
%template(SeeableInference) shogun::Seedable<shogun::Inference>;
%template(RandomMixinInference) shogun::RandomMixin<shogun::Inference, std::mt19937_64>;
%include <shogun/machine/gp/LaplaceInference.h>
%include <shogun/machine/gp/SparseInference.h>
%include <shogun/machine/gp/SingleSparseInference.h>
%include <shogun/machine/gp/SingleFITCInference.h>
%include <shogun/machine/gp/SingleLaplaceInferenceMethod.h>
%include <shogun/machine/gp/MultiLaplaceInferenceMethod.h>
%include <shogun/machine/gp/ExactInferenceMethod.h>
%include <shogun/machine/gp/SingleFITCLaplaceInferenceMethod.h>
%include <shogun/machine/gp/FITCInferenceMethod.h>
%include <shogun/machine/gp/VarDTCInferenceMethod.h>
%include <shogun/machine/gp/EPInferenceMethod.h>

%include <shogun/machine/gp/KLInference.h>
%include <shogun/machine/gp/KLLowerTriangularInference.h>
%include <shogun/machine/gp/KLCovarianceInferenceMethod.h>
%include <shogun/machine/gp/KLDiagonalInferenceMethod.h>
%include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
%include <shogun/machine/gp/KLDualInferenceMethod.h>

%include <shogun/machine/GaussianProcessMachine.h>
%include <shogun/classifier/GaussianProcessClassification.h>
%include <shogun/regression/GaussianProcessRegression.h>
