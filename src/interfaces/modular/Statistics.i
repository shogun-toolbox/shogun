/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

/* Remove C Prefix */
%rename(HypothesisTest) CHypothesisTest;
%rename(IndependenceTest) CIndependenceTest;
%rename(TwoSampleTest) CTwoSampleTest;
%rename(KernelTwoSampleTest) CKernelTwoSampleTest;
%rename(StreamingMMD) CStreamingMMD;
%rename(LinearTimeMMD) CLinearTimeMMD;
%rename(BTestMMD) CBTestMMD;
%rename(QuadraticTimeMMD) CQuadraticTimeMMD;
%rename(KernelIndependenceTest) CKernelIndependenceTest;
%rename(HSIC) CHSIC;
%rename(KernelMeanMatching) CKernelMeanMatching;
%rename(MMDKernelSelection) CMMDKernelSelection;
%rename(MMDKernelSelectionComb) CMMDKernelSelectionComb;
%rename(MMDKernelSelectionMedian) CMMDKernelSelectionMedian;
%rename(MMDKernelSelectionMax) CMMDKernelSelectionMax;
%rename(MMDKernelSelectionOpt) CMMDKernelSelectionOpt;
%rename(MMDKernelSelectionCombOpt) CMMDKernelSelectionCombOpt;
%rename(MMDKernelSelectionCombMaxL2) CMMDKernelSelectionCombMaxL2;


/* Include Class Headers to make them visible from within the target language */
%include <shogun/statistics/HypothesisTest.h>
%include <shogun/statistics/IndependenceTest.h>
%include <shogun/statistics/TwoSampleTest.h>
%include <shogun/statistics/KernelTwoSampleTest.h>
%include <shogun/statistics/StreamingMMD.h>
%include <shogun/statistics/LinearTimeMMD.h>
%include <shogun/statistics/BTestMMD.h>
%include <shogun/statistics/QuadraticTimeMMD.h>
%include <shogun/statistics/KernelIndependenceTest.h>
%include <shogun/statistics/HSIC.h>
%include <shogun/statistics/KernelMeanMatching.h>
%include <shogun/statistics/MMDKernelSelection.h>
%include <shogun/statistics/MMDKernelSelectionComb.h>
%include <shogun/statistics/MMDKernelSelectionMedian.h>
%include <shogun/statistics/MMDKernelSelectionMax.h>
%include <shogun/statistics/MMDKernelSelectionOpt.h>
%include <shogun/statistics/MMDKernelSelectionCombOpt.h>
%include <shogun/statistics/MMDKernelSelectionCombMaxL2.h>
