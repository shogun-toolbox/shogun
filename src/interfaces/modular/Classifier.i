/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_w(self) -> [] of float") get_w;
%feature("autodoc", "get_support_vectors(self) -> [] of int") get_support_vectors;
%feature("autodoc", "get_alphas(self) -> [] of float") get_alphas;
#endif

/* Remove C Prefix */
%rename(Machine) CMachine;
%rename(KernelMachine) CKernelMachine;
%rename(GNPPSVM) CGNPPSVM;
%rename(GPBTSVM) CGPBTSVM;
%rename(GaussianNaiveBayes) CGaussianNaiveBayes;
%rename(LDA) CLDA;
%rename(QDA) CQDA;
%rename(LibLinear) CLibLinear;
%rename(LibSVM) CLibSVM;
%rename(LibSVMOneClass) CLibSVMOneClass;
%rename(LinearMachine) CLinearMachine;
%rename(OnlineLinearMachine) COnlineLinearMachine;
%rename(LPBoost) CLPBoost;
%rename(LPM) CLPM;
%rename(MPDSVM) CMPDSVM;
%rename(OnlineSVMSGD) COnlineSVMSGD;
%rename(OnlineLibLinear) COnlineLibLinear;
%rename(Perceptron) CPerceptron;
%rename(AveragedPerceptron) CAveragedPerceptron;
%rename(SubGradientLPM) CSubGradientLPM;
%rename(SubGradientSVM) CSubGradientSVM;
%rename(NewtonSVM) CNewtonSVM;
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
%rename(VowpalWabbit) CVowpalWabbit;
#ifdef USE_SVMLIGHT
%rename(SVMLight) CSVMLight;
#endif //USE_SVMLIGHT

%rename(ThresholdRejectionStrategy) CThresholdRejectionStrategy;
%rename(DixonQTestRejectionStrategy) CDixonQTestRejectionStrategy;
%rename(MulticlassStrategy) CMulticlassStrategy;
%rename(MulticlassOneVsRestStrategy) CMulticlassOneVsRestStrategy;
%rename(MulticlassOneVsOneStrategy) CMulticlassOneVsOneStrategy;
%rename(KernelMulticlassMachine) CKernelMulticlassMachine;
%rename(LinearMulticlassMachine) CLinearMulticlassMachine;
%rename(MulticlassSVM) CMulticlassSVM;
%rename(MKLMulticlass) CMKLMulticlass;

%rename(ECOCStrategy) CECOCStrategy;
%rename(ECOCEncoder) CECOCEncoder;
%rename(ECOCDecoder) CECOCDecoder;
%rename(ECOCOVREncoder) CECOCOVREncoder;
%rename(ECOCHDDecoder) CECOCHDDecoder;

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
%include <shogun/classifier/GaussianNaiveBayes.h>
%include <shogun/classifier/svm/GNPPSVM.h>
%include <shogun/classifier/svm/GPBTSVM.h>
%include <shogun/classifier/LDA.h>
%include <shogun/classifier/QDA.h>
%include <shogun/classifier/svm/LibLinear.h>
%include <shogun/classifier/svm/LibSVM.h>
%include <shogun/classifier/svm/LibSVMOneClass.h>
%include <shogun/classifier/LPBoost.h> 
%include <shogun/classifier/LPM.h>
%include <shogun/classifier/svm/MPDSVM.h>
%include <shogun/classifier/svm/OnlineSVMSGD.h>
%include <shogun/classifier/svm/OnlineLibLinear.h>
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
%include <shogun/classifier/vw/VowpalWabbit.h>
%include <shogun/classifier/svm/NewtonSVM.h>

%include <shogun/multiclass/RejectionStrategy.h>
%include <shogun/multiclass/MulticlassStrategy.h>
%include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
%include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
%include <shogun/machine/MulticlassMachine.h>
%include <shogun/machine/LinearMulticlassMachine.h>
%include <shogun/machine/KernelMulticlassMachine.h>
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/classifier/mkl/MKLMulticlass.h>

%include <shogun/multiclass/ecoc/ECOCStrategy.h>
%include <shogun/multiclass/ecoc/ECOCEncoder.h>
%include <shogun/multiclass/ecoc/ECOCDecoder.h>
%include <shogun/multiclass/ecoc/ECOCOVREncoder.h>
%include <shogun/multiclass/ecoc/ECOCHDDecoder.h>

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
