/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

/* Remove C Prefix */
%rename(RejectionStrategy) CRejectionStrategy;
%rename(ThresholdRejectionStrategy) CThresholdRejectionStrategy;
%rename(DixonQTestRejectionStrategy) CDixonQTestRejectionStrategy;
%rename(MulticlassStrategy) CMulticlassStrategy;
%rename(MulticlassOneVsRestStrategy) CMulticlassOneVsRestStrategy;
%rename(MulticlassOneVsOneStrategy) CMulticlassOneVsOneStrategy;
%rename(MulticlassMachine) CMulticlassMachine;
%rename(LinearMulticlassMachine) CLinearMulticlassMachine;
%rename(KernelMulticlassMachine) CKernelMulticlassMachine;
%rename(MulticlassSVM) CMulticlassSVM;
%rename(MKLMulticlass) CMKLMulticlass;

%rename(ECOCStrategy) CECOCStrategy;
%rename(ECOCEncoder) CECOCEncoder;
%rename(ECOCDecoder) CECOCDecoder;
%rename(ECOCOVREncoder) CECOCOVREncoder;
%rename(ECOCOVOEncoder) CECOCOVOEncoder;
%rename(ECOCRandomSparseEncoder) CECOCRandomSparseEncoder;
%rename(ECOCRandomDenseEncoder) CECOCRandomDenseEncoder;
%rename(ECOCHDDecoder) CECOCHDDecoder;

%rename(MulticlassLibLinear) CMulticlassLibLinear;
%rename(MulticlassOCAS) CMulticlassOCAS;
%rename(MulticlassSVM) CMulticlassSVM;
%rename(MulticlassLibSVM) CMulticlassLibSVM;
%rename(LaRank) CLaRank;
%rename(ScatterSVM) CScatterSVM;
%rename(GMNPSVM) CGMNPSVM;
%rename(KNN) CKNN;
%rename(ConjugateIndex) CConjugateIndex;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/multiclass/RejectionStrategy.h>
%include <shogun/multiclass/MulticlassStrategy.h>
%include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
%include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
%include <shogun/machine/MulticlassMachine.h>
%include <shogun/machine/LinearMulticlassMachine.h>
%include <shogun/machine/KernelMulticlassMachine.h>
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/classifier/mkl/MKLMulticlass.h>

%include <shogun/multiclass/ecoc/ECOCEncoder.h>
%include <shogun/multiclass/ecoc/ECOCDecoder.h>
%include <shogun/multiclass/ecoc/ECOCOVREncoder.h>
%include <shogun/multiclass/ecoc/ECOCOVOEncoder.h>
%include <shogun/multiclass/ecoc/ECOCRandomSparseEncoder.h>
%include <shogun/multiclass/ecoc/ECOCRandomDenseEncoder.h>
%include <shogun/multiclass/ecoc/ECOCHDDecoder.h>
%include <shogun/multiclass/ecoc/ECOCStrategy.h>

%include <shogun/multiclass/MulticlassLibLinear.h>
%include <shogun/multiclass/MulticlassOCAS.h>
%include <shogun/multiclass/MulticlassSVM.h>
%include <shogun/multiclass/MulticlassLibSVM.h>
%include <shogun/multiclass/LaRank.h>
%include <shogun/multiclass/ScatterSVM.h>
%include <shogun/multiclass/GMNPSVM.h>
%include <shogun/multiclass/KNN.h>
%include <shogun/multiclass/ConjugateIndex.h>

