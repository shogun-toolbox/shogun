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
%rename(TreeBasedGroupLassoLinearRegression) CTreeBasedGroupLassoLinearRegression;
%rename(MultitaskKernelNormalizer) CMultitaskKernelNormalizer;
%rename(MultitaskKernelMklNormalizer) CMultitaskKernelMklNormalizer;
%rename(MultitaskKernelTreeNormalizer) CMultitaskKernelTreeNormalizer;
%rename(MultitaskKernelMaskNormalizer) CMultitaskKernelMaskNormalizer;
%rename(MultitaskKernelMaskPairNormalizer) CMultitaskKernelMaskPairNormalizer;
%rename(MultitaskKernelPlifNormalizer) CMultitaskKernelPlifNormalizer;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/multitask/TreeBasedGroupLassoLinearRegression.h>
%include <shogun/multitask/MultitaskKernelNormalizer.h>
%include <shogun/multitask/MultitaskKernelNormalizer.h>
%include <shogun/multitask/MultitaskKernelTreeNormalizer.h>
%include <shogun/multitask/MultitaskKernelMaskNormalizer.h>
%include <shogun/multitask/MultitaskKernelMaskPairNormalizer.h>
%include <shogun/multitask/MultitaskKernelPlifNormalizer.h>

