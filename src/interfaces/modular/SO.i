/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) Fernando José Iglesias García
 * Copyright (C) Fernando José Iglesias García
 */

/* SO renames */
%rename(StructuredOutputMachine) CStructuredOutputMachine;
%rename(LinearStructuredOutputMachine) CLinearStructuredOutputMachine;
%rename(StructuredModel) CStructuredModel;
%rename(MulticlassModel) CMulticlassModel;
%rename(MulticlassSOLabels) CMulticlassSOLabels;

#ifdef USE_MOSEK
%rename(PrimalMosekSOSVM) CPrimalMosekSOSVM;
#endif /* USE_MOSEK */

/* SO includes */
%include <shogun/machine/StructuredOutputMachine.h>
%include <shogun/machine/LinearStructuredOutputMachine.h>

%include <shogun/so/StructuredModel.h>
%include <shogun/so/MulticlassModel.h>
%include <shogun/so/MulticlassSOLabels.h>

#ifdef USE_MOSEK
%include <shogun/so/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
