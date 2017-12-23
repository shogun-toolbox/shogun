%{
 #include <shogun/optimization/Minimizer.h>
 #include <shogun/optimization/FirstOrderMinimizer.h>
 #include <shogun/optimization/lbfgs/lbfgscommon.h>
 #include <shogun/optimization/lbfgs/LBFGSMinimizer.h>
#ifdef USE_GPL_SHOGUN
#ifdef HAVE_NLOPT
 #include <shogun/optimization/nloptcommon.h>
 #include <shogun/optimization/NLOPTMinimizer.h>
#endif //HAVE_NLOPT
#endif //USE_GPL_SHOGUN
%}
