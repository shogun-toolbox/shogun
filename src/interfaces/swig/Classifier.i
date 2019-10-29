/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Saloni Nigam, Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
%feature("autodoc", "get_support_vectors(self) -> [] of int") get_support_vectors;
%feature("autodoc", "get_alphas(self) -> [] of float") get_alphas;
#endif

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::DirectorLinearMachine;
%feature("director") shogun::DirectorKernelMachine;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

/* Remove C Prefix */
%shared_ptr(shogun::Machine)
%shared_ptr(shogun::KernelMachine)
%shared_ptr(shogun::SVM)
%shared_ptr(shogun::GNPPSVM)
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::GPBTSVM)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::LDA)
%shared_ptr(shogun::DenseRealDispatch<shogun::LDA, shogun::LinearMachine>)
%shared_ptr(shogun::LinearMachine)
%shared_ptr(shogun::IterativeMachine<shogun::LinearMachine>)
%shared_ptr(shogun::OnlineLinearMachine)
%shared_ptr(shogun::LPBoost)
%shared_ptr(shogun::LPM)
%shared_ptr(shogun::MPDSVM)
%shared_ptr(shogun::OnlineSVMSGD)
%shared_ptr(shogun::Perceptron)
%shared_ptr(shogun::AveragedPerceptron)
#ifndef HAVE_PYTHON
%shared_ptr(shogun::SVM)
#endif
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::SVMLin)
%shared_ptr(shogun::SVMOcas)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::SVMSGD)
%shared_ptr(shogun::SGDQN)
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::WDSVMOcas)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::PluginEstimate)
%shared_ptr(shogun::MKL)
%shared_ptr(shogun::MKLClassification)
%shared_ptr(shogun::MKLOneClass)
%shared_ptr(shogun::VowpalWabbit)
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::FeatureBlockLogisticRegression)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::DirectorLinearMachine)
%shared_ptr(shogun::DirectorKernelMachine)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/Machine.h>
%include <shogun/machine/IterativeMachine.h>
%include <shogun/machine/FeatureDispatchCRTP.h>
%include <shogun/machine/KernelMachine.h>
%include <shogun/machine/DistanceMachine.h>
%include <shogun/classifier/svm/SVM.h>
%include <shogun/machine/LinearMachine.h>
%template(LinearIterativeMachine) shogun::IterativeMachine<shogun::LinearMachine>;
%include <shogun/machine/OnlineLinearMachine.h>
%include <shogun/classifier/svm/GNPPSVM.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/classifier/svm/GPBTSVM.h>
#endif //USE_GPL_SHOGUN
%include <shogun/classifier/LDA.h>
%template(DenseDispatchLDA) shogun::DenseRealDispatch<shogun::LDA, shogun::LinearMachine>;

%include <shogun/classifier/LPBoost.h>
%include <shogun/classifier/LPM.h>
%include <shogun/classifier/svm/MPDSVM.h>
%include <shogun/classifier/svm/OnlineSVMSGD.h>
%include <shogun/classifier/Perceptron.h>
%include <shogun/classifier/AveragedPerceptron.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/classifier/svm/SVMLin.h>
%include <shogun/classifier/svm/SVMOcas.h>
%include <shogun/classifier/svm/SVMSGD.h>
#endif //USE_GPL_SHOGUN
%include <shogun/classifier/svm/SGDQN.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/classifier/svm/WDSVMOcas.h>
#endif //USE_GPL_SHOGUN
%include <shogun/classifier/PluginEstimate.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/classifier/mkl/MKLClassification.h>
%include <shogun/classifier/mkl/MKLOneClass.h>

#ifdef USE_GPL_SHOGUN
%include <shogun/classifier/FeatureBlockLogisticRegression.h>
#endif //USE_GPL_SHOGUN
%include <shogun/machine/DirectorLinearMachine.h>
%include <shogun/machine/DirectorKernelMachine.h>
