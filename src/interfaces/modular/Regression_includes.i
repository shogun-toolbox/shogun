%{
 #include <regression/Regression.h>
 #include <machine/Machine.h>
 #include <machine/KernelMachine.h>
 #include <regression/GaussianProcessRegression.h>
 #include <regression/KernelRidgeRegression.h>
 #include <regression/LinearRidgeRegression.h>
 #include <regression/LeastSquaresRegression.h>
 #include <regression/LeastAngleRegression.h>
 #include <classifier/svm/SVM.h>
 #include <classifier/svm/LibSVM.h>
 #include <regression/svr/LibSVR.h>
 #include <regression/svr/LibLinearRegression.h>
 #include <classifier/mkl/MKL.h>
 #include <regression/svr/MKLRegression.h>
#ifdef USE_SVMLIGHT
 #include <classifier/svm/SVMLight.h>
 #include <regression/svr/SVRLight.h>
#endif //USE_SVMLIGHT
%}
