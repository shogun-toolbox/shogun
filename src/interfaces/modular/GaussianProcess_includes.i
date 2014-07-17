%{
#ifdef HAVE_EIGEN3

 #include <shogun/evaluation/DifferentiableFunction.h>
 #include <shogun/machine/gp/LikelihoodModel.h>
 #include <shogun/machine/gp/ProbitLikelihood.h>
 #include <shogun/machine/gp/LogitLikelihood.h>
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
 #include <shogun/machine/gp/LaplacianInferenceMethod.h>
 #include <shogun/machine/gp/ExactInferenceMethod.h>
 #include <shogun/machine/gp/LaplacianInferenceMethodWithLBFGS.h>
 #include <shogun/machine/gp/FITCInferenceMethod.h>
 #include <shogun/machine/gp/EPInferenceMethod.h>

 #include <shogun/machine/gp/KLInferenceMethod.h>
 #include <shogun/machine/gp/KLLowerTriangularInferenceMethod.h>
 #include <shogun/machine/gp/KLFullDiagonalInferenceMethod.h>
 #include <shogun/machine/gp/KLApproxDiagonalInferenceMethod.h>
 #include <shogun/machine/gp/KLCholeskyInferenceMethod.h>
 #include <shogun/machine/gp/KLDualInferenceMethod.h>

 #include <shogun/machine/GaussianProcessMachine.h>
 #include <shogun/classifier/GaussianProcessBinaryClassification.h>
 #include <shogun/regression/GaussianProcessRegression.h>
#endif //HAVE_EIGEN3
%}
