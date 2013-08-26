%{
#ifdef HAVE_EIGEN3

 #include <shogun/evaluation/DifferentiableFunction.h>
 #include <shogun/machine/gp/LikelihoodModel.h>
 #include <shogun/machine/gp/ProbitLikelihood.h>
 #include <shogun/machine/gp/LogitLikelihood.h>
 #include <shogun/machine/gp/GaussianLikelihood.h>
 #include <shogun/machine/gp/StudentsTLikelihood.h>

 #include <shogun/machine/gp/MeanFunction.h>
 #include <shogun/machine/gp/ZeroMean.h>

 #include <shogun/machine/gp/InferenceMethod.h>
 #include <shogun/machine/gp/LaplacianInferenceMethod.h>
 #include <shogun/machine/gp/ExactInferenceMethod.h>
 #include <shogun/machine/gp/LaplacianInferenceMethod.h>
 #include <shogun/machine/gp/FITCInferenceMethod.h>
 #include <shogun/machine/gp/EPInferenceMethod.h>

 #include <shogun/machine/gp/MeanFunction.h>
 #include <shogun/machine/gp/ZeroMean.h>

 #include <shogun/machine/GaussianProcessMachine.h>
 #include <shogun/classifier/GaussianProcessBinaryClassification.h>
 #include <shogun/regression/GaussianProcessRegression.h>
#endif //HAVE_EIGEN3
%}
