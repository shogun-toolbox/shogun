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
%rename(StructuredModel) CStructuredModel;
%rename(StructuredLossFunction) CStructuredLossFunction;
%rename(ArgMaxFunction) CArgMaxFunction;

#ifdef USE_MOSEK
%rename(PrimalMosekSOSVM) CPrimalMosekSOSVM;
#endif /* USE_MOSEK */

/* SO includes */
%include <shogun/so/StructuredModel.h>
%include <shogun/so/StructuredLossFunction.h>
%include <shogun/so/ArgMaxFunction.h>

#ifdef USE_MOSEK
%include <shogun/so/PrimalMosekSOSVM.h>
#endif /* USE_MOSEK */
