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
%feature("director") shogun::CDirectorLinearMachine;
%feature("director") shogun::CDirectorKernelMachine;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

/* Remove C Prefix */
%rename(Machine) CMachine;
%rename(KernelMachine) CKernelMachine;
%rename(GNPPSVM) CGNPPSVM;
#ifdef USE_GPL_SHOGUN
%rename(GPBTSVM) CGPBTSVM;
#endif //USE_GPL_SHOGUN
%rename(LDA) CLDA;
#ifdef USE_SVMLIGHT
%rename(SVMLight) CSVMLight;
%rename(SVMLightOneClass) CSVMLightOneClass;
#endif //USE_SVMLIGHT
%rename(LinearMachine) CLinearMachine;
%rename(OnlineLinearMachine) COnlineLinearMachine;
%rename(LPBoost) CLPBoost;
%rename(LPM) CLPM;
%rename(MPDSVM) CMPDSVM;
%rename(OnlineSVMSGD) COnlineSVMSGD;
%rename(Perceptron) CPerceptron;
%rename(AveragedPerceptron) CAveragedPerceptron;
#ifndef HAVE_PYTHON
%rename(SVM) CSVM;
#endif
#ifdef USE_GPL_SHOGUN
%rename(SVMLin) CSVMLin;
%rename(SVMOcas) CSVMOcas;
#endif //USE_GPL_SHOGUN
%rename(SVMSGD) CSVMSGD;
%rename(SGDQN) CSGDQN;
#ifdef USE_GPL_SHOGUN
%rename(WDSVMOcas) CWDSVMOcas;
#endif //USE_GPL_SHOGUN
%rename(PluginEstimate) CPluginEstimate;
%rename(MKL) CMKL;
%rename(MKLClassification) CMKLClassification;
%rename(MKLOneClass) CMKLOneClass;
%rename(VowpalWabbit) CVowpalWabbit;
#ifdef USE_GPL_SHOGUN
%rename(FeatureBlockLogisticRegression) CFeatureBlockLogisticRegression;
#endif //USE_GPL_SHOGUN
%rename(DirectorLinearMachine) CDirectorLinearMachine;
%rename(DirectorKernelMachine) CDirectorKernelMachine;
%rename(BaggingMachine) CBaggingMachine;

/* These functions return new Objects */
%newobject apply();
%newobject apply(CFeatures* data);
%newobject apply_locked(const SGVector<index_t>& indices);
%newobject classify();
%newobject classify(CFeatures* data);

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/Machine.h>
%include <shogun/machine/KernelMachine.h>
%include <shogun/machine/DistanceMachine.h>
%include <shogun/classifier/svm/SVM.h>
%include <shogun/machine/LinearMachine.h>
%include <shogun/machine/OnlineLinearMachine.h>
%include <shogun/classifier/svm/GNPPSVM.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/classifier/svm/GPBTSVM.h>
#endif //USE_GPL_SHOGUN
%include <shogun/classifier/LDA.h>
#ifdef USE_SVMLIGHT
%ignore VERSION;
%ignore VERSION_DATE;
%ignore MAXSHRINK;
%ignore SHRINK_STATE;
%ignore MODEL;
%ignore LEARN_PARM;
%ignore TIMING;

%include <shogun/classifier/svm/SVMLight.h>
%include <shogun/classifier/svm/SVMLightOneClass.h>
#endif //USE_SVMLIGHT

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
%include <shogun/machine/BaggingMachine.h>
