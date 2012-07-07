%{
 #include <shogun/regression/Regression.h>
 #include <shogun/machine/Machine.h>
 #include <shogun/machine/KernelMachine.h>
 #include <shogun/regression/KernelRidgeRegression.h>
 #include <shogun/regression/LinearRidgeRegression.h>
 #include <shogun/regression/LeastSquaresRegression.h>
 #include <shogun/regression/gp/LikelihoodModel.h>
 #include <shogun/regression/gp/GaussianLikelihood.h>
 #include <shogun/regression/gp/MeanFunction.h>
 #include <shogun/regression/gp/ZeroMean.h>
 #include <shogun/regression/gp/InferenceMethod.h>
 #include <shogun/regression/gp/ExactInferenceMethod.h>
 #include <shogun/regression/GaussianProcessRegression.h>
 #include <shogun/regression/LeastAngleRegression.h>
 #include <shogun/classifier/svm/SVM.h>
 #include <shogun/classifier/svm/LibSVM.h>
 #include <shogun/regression/svr/LibSVR.h>
 #include <shogun/regression/svr/LibLinearRegression.h>
 #include <shogun/classifier/mkl/MKL.h>
 #include <shogun/regression/svr/MKLRegression.h>
 #include <shogun/machine/SLEPMachine.h>
#ifdef USE_SVMLIGHT
 #include <shogun/classifier/svm/SVMLight.h>
 #include <shogun/regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT
%}
