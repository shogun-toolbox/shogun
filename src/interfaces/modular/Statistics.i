/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */
 
/* Remove C Prefix */
%rename(StatisticalTest) CStatisticalTest;
%rename(TestStatistic) CTestStatistic;
%rename(TwoSampleTestStatistic) CTwoSampleTestStatistic;
%rename(KernelTwoSampleTestStatistic) CKernelTwoSampleTestStatistic;
%rename(LinearTimeMMD) CLinearTimeMMD;
%rename(QuadraticTimeMMD) CQuadraticTimeMMD;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/statistics/StatisticalTest.h>
%include <shogun/statistics/TestStatistic.h>
%include <shogun/statistics/TwoSampleTestStatistic.h>
%include <shogun/statistics/KernelTwoSampleTestStatistic.h>
%include <shogun/statistics/LinearTimeMMD.h>
%include <shogun/statistics/QuadraticTimeMMD.h>
