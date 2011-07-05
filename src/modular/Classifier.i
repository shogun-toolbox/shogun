/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

%define DOCSTR
"The `Classifier` module gathers all classifiers available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Classifier
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Classifier_doxygen.i"
#endif
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
%feature("autodoc", "get_support_vectors(self) -> [] of int") get_support_vectors;
%feature("autodoc", "get_alphas(self) -> [] of float") get_alphas;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Features_includes.i"
%include "Kernel_includes.i"
%include "Distance_includes.i"
%include "Classifier_includes.i"
%include "Preprocessor_includes.i"
%include "Library_includes.i"

%import "Features.i"
%import "Kernel.i"
%import "Distance.i"

/* Remove C Prefix */
/*%rename(Labels) CLabels;*/
%rename(Machine) CMachine;
%rename(KernelMachine) CKernelMachine;
%rename(GMNPSVM) CGMNPSVM;
%rename(GNPPSVM) CGNPPSVM;
%rename(GPBTSVM) CGPBTSVM;
%rename(GaussianNaiveBayes) CGaussianNaiveBayes;
%rename(KernelPerceptron) CKernelPerceptron;
%rename(KNN) CKNN;
%rename(LDA) CLDA;
%rename(LibLinear) CLibLinear;
%rename(ScatterSVM) CScatterSVM;
%rename(LibSVM) CLibSVM;
%rename(LaRank) CLaRank;
%rename(LibSVMMultiClass) CLibSVMMultiClass;
%rename(LibSVMOneClass) CLibSVMOneClass;
%rename(LinearMachine) CLinearMachine;
%rename(LPBoost) CLPBoost;
%rename(LPM) CLPM;
%rename(MPDSVM) CMPDSVM;
%rename(MultiClassSVM) CMultiClassSVM;
%rename(Perceptron) CPerceptron;
%rename(AveragedPerceptron) CAveragedPerceptron;
%rename(SubGradientLPM) CSubGradientLPM;
%rename(SubGradientSVM) CSubGradientSVM;
#ifndef HAVE_PYTHON
%rename(SVM) CSVM;
#endif
%rename(SVMLin) CSVMLin;
%rename(SVMOcas) CSVMOcas;
%rename(SVMSGD) CSVMSGD;
%rename(SGDQN) CSGDQN;
%rename(WDSVMOcas) CWDSVMOcas;
%rename(PluginEstimate) CPluginEstimate;
%rename(MKL) CMKL;
%rename(MKLClassification) CMKLClassification;
%rename(MKLOneClass) CMKLOneClass;
%rename(MKLMultiClass) CMKLMultiClass;
#ifdef USE_SVMLIGHT
%rename(SVMLight) CSVMLight;
%rename(DomainAdaptationSVM) CDomainAdaptationSVM;
%rename(DomainAdaptationSVMLinear) CDomainAdaptationSVMLinear;
#endif //USE_SVMLIGHT

/* These functions return new Objects */
%newobject classify();
%newobject classify(CFeatures* data);

/* Include Class Headers to make them visible from within the target language */
/*%include <shogun/features/Labels.h>*/
%include <shogun/machine/Machine.h>
%include <shogun/machine/KernelMachine.h>
%include <shogun/machine/DistanceMachine.h>
%include <shogun/classifier/svm/SVM.h>
%include <shogun/classifier/svm/MultiClassSVM.h>
%include <shogun/machine/LinearMachine.h> 
%include <shogun/classifier/GaussianNaiveBayes.h>
%include <shogun/classifier/svm/GMNPSVM.h>
%include <shogun/classifier/svm/GNPPSVM.h>
%include <shogun/classifier/svm/GPBTSVM.h>
%include <shogun/classifier/KernelPerceptron.h> 
%include <shogun/classifier/KNN.h>
%include <shogun/classifier/LDA.h>
%include <shogun/classifier/svm/LibLinear.h>
%include <shogun/classifier/svm/ScatterSVM.h>
%include <shogun/classifier/svm/LibSVM.h>
%include <shogun/classifier/svm/LaRank.h>
%include <shogun/classifier/svm/LibSVMMultiClass.h>
%include <shogun/classifier/svm/LibSVMOneClass.h>
%include <shogun/classifier/LPBoost.h> 
%include <shogun/classifier/LPM.h>
%include <shogun/classifier/svm/MPDSVM.h>
%include <shogun/classifier/Perceptron.h>
%include <shogun/classifier/AveragedPerceptron.h>
%include <shogun/classifier/SubGradientLPM.h>
%include <shogun/classifier/svm/SubGradientSVM.h>
%include <shogun/classifier/svm/SVMLin.h>
%include <shogun/classifier/svm/SVMOcas.h>
%include <shogun/classifier/svm/SVMSGD.h>
%include <shogun/classifier/svm/SGDQN.h>
%include <shogun/classifier/svm/WDSVMOcas.h>
%include <shogun/classifier/PluginEstimate.h> 
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/classifier/mkl/MKLClassification.h>
%include <shogun/classifier/mkl/MKLOneClass.h>
%include <shogun/classifier/mkl/MKLMultiClass.h>
%include <shogun/classifier/svm/DomainAdaptationSVMLinear.h>

#ifdef HAVE_PYTHON
%pythoncode %{
  class SVM(CSVM):
      def __init__(self, kernel, alphas, support_vectors, b):
          assert(len(alphas)==len(support_vectors))
          num_sv=len(alphas)
          CSVM.__init__(self, num_sv)
          self.set_alphas(alphas)
          self.set_support_vectors(support_vectors)
          self.set_kernel(kernel)
          self.set_bias(b)
%}
#endif //HAVE_PYTHON


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
%include <shogun/classifier/svm/DomainAdaptationSVM.h>

#endif //USE_SVMLIGHT
