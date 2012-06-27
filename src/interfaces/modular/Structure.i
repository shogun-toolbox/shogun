/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg, 2012 Fernando José Iglesias García
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
/* Remove C Prefix */
%rename(PlifBase) CPlifBase;
%rename(Plif) CPlif;
%rename(PlifArray) CPlifArray;
%rename(DynProg) CDynProg;
%rename(PlifMatrix) CPlifMatrix;
%rename(SegmentLoss) CSegmentLoss;
%rename(IntronList) CIntronList;

%rename(StructuredOutputMachine) CStructuredOutputMachine;
%rename(LinearStructuredOutputMachine) CLinearStructuredOutputMachine;
%rename(KernelStructuredOutputMachine) CKernelStructuredOutputMachine;
%rename(StructuredModel) CStructuredModel;
%rename(MulticlassModel) CMulticlassModel;
%rename(MulticlassSOLabels) CMulticlassSOLabels;
%rename(RealNumber) CRealNumber;

#ifdef USE_MOSEK
%rename(PrimalMosekSOSVM) CPrimalMosekSOSVM;
#endif /* USE_MOSEK */

/* Include Class Headers to make them visible from within the target language */
%include <shogun/structure/PlifBase.h>
%include <shogun/structure/Plif.h>
%include <shogun/structure/PlifArray.h>
%include <shogun/structure/DynProg.h>
%include <shogun/structure/PlifMatrix.h>
%include <shogun/structure/IntronList.h>
%include <shogun/structure/SegmentLoss.h>

%include <shogun/machine/StructuredOutputMachine.h>
%include <shogun/machine/LinearStructuredOutputMachine.h>
%include <shogun/machine/KernelStructuredOutputMachine.h>

%include <shogun/structure/StructuredModel.h>
%include <shogun/structure/MulticlassModel.h>
%include <shogun/structure/MulticlassSOLabels.h>

#ifdef USE_MOSEK
%include <shogun/structure/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
