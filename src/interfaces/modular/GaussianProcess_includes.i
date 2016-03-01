%{

 #include <shogun/evaluation/DifferentiableFunction.h>
 #include <shogun/machine/gp/LikelihoodModel.h>
 #include <shogun/machine/gp/ProbitLikelihood.h>
 #include <shogun/machine/gp/LogitLikelihood.h>
 #include <shogun/machine/gp/SoftMaxLikelihood.h>
 #include <shogun/machine/gp/GaussianLikelihood.h>
 #include <shogun/machine/gp/StudentsTLikelihood.h>

 #include <shogun/machine/gp/VariationalLikelihood.h>
 #include <shogun/machine/gp/VariationalGaussianLikelihood.h>
 #include <shogun/machine/gp/NumericalVGLikelihood.h>
 #include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>
 #include <shogun/machine/gp/LogitVGLikelihood.h>
 #include <shogun/machine/gp/LogitVGPiecewiseBoundLikelihood.h>
 #include <shogun/machine/gp/LogitDVGLikelihood.h>
 #include <shogun/machine/gp/ProbitVGLikelihood.h>
 #include <shogun/machine/gp/StudentsTVGLikelihood.h>

 #include <shogun/machine/gp/MeanFunction.h>
 #include <shogun/machine/gp/ZeroMean.h>
 #include <shogun/machine/gp/ConstMean.h>

 #include <shogun/machine/gp/InferenceMethod.h>
 #include <shogun/machine/gp/LaplacianInferenceBase.h>
 #include <shogun/machine/gp/SparseInferenceBase.h>
 #include <shogun/machine/gp/SingleFITCLaplacianBase.h>
 #include <shogun/machine/gp/SingleLaplacianInferenceMethod.h>
 #include <shogun/machine/gp/SingleSparseInferenceBase.h>
 #include <shogun/machine/gp/MultiLaplacianInferenceMethod.h>
 #include <shogun/machine/gp/ExactInferenceMethod.h>
 #include <shogun/machine/gp/SingleLaplacianInferenceMethodWithLBFGS.h>
 #include <shogun/machine/gp/FITCInferenceMethod.h>
 #include <shogun/machine/gp/SingleFITCLaplacianInferenceMethod.h>
 #include <shogun/machine/gp/SingleFITCLaplacianInferenceMethodWithLBFGS.h>
 #include <shogun/machine/gp/EPInferenceMethod.h>

 #include <shogun/machine/gp/KLInferenceMethod.h>
 #include <shogun/machine/gp/KLLowerTriangularInferenceMethod.h>
 #include <shogun/machine/gp/KLCovarianceInferenceMethod.h>
 #include <shogun/machine/gp/KLApproxDiagonalInferenceMethod.h>
 #include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
 #include <shogun/machine/gp/KLDualInferenceMethod.h>

 #include <shogun/machine/GaussianProcessMachine.h>
 #include <shogun/classifier/GaussianProcessClassification.h>
 #include <shogun/regression/GaussianProcessRegression.h>
%}
