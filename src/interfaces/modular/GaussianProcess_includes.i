%{
#ifdef HAVE_EIGEN3

 #include <evaluation/DifferentiableFunction.h>
 #include <machine/gp/LikelihoodModel.h>
 #include <machine/gp/ProbitLikelihood.h>
 #include <machine/gp/LogitLikelihood.h>
 #include <machine/gp/GaussianLikelihood.h>
 #include <machine/gp/StudentsTLikelihood.h>

 #include <machine/gp/MeanFunction.h>
 #include <machine/gp/ZeroMean.h>

 #include <machine/gp/InferenceMethod.h>
 #include <machine/gp/LaplacianInferenceMethod.h>
 #include <machine/gp/ExactInferenceMethod.h>
 #include <machine/gp/LaplacianInferenceMethod.h>
 #include <machine/gp/FITCInferenceMethod.h>
 #include <machine/gp/EPInferenceMethod.h>

 #include <machine/gp/MeanFunction.h>
 #include <machine/gp/ZeroMean.h>

 #include <machine/GaussianProcessMachine.h>
 #include <classifier/GaussianProcessBinaryClassification.h>
 #include <regression/GaussianProcessRegression.h>
#endif //HAVE_EIGEN3
%}
