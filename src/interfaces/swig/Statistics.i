/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* These functions return new Objects */
%newobject shogun::CTwoDistributionTest::compute_distance(CDistance*);
%newobject shogun::CTwoDistributionTest::compute_joint_distance(CDistance*);
%newobject shogun::CQuadraticTimeMMD::get_p_and_q();

/* Remove C Prefix */
%rename(HypothesisTest) CHypothesisTest;
%rename(OneDistributionTest) COneDistributionTest;
%rename(TwoDistributionTest) CTwoDistributionTest;
%rename(IndependenceTest) CIndependenceTest;
%rename(TwoSampleTest) CTwoSampleTest;
%rename(MMD) CMMD;
%rename(StreamingMMD) CStreamingMMD;
%rename(LinearTimeMMD) CLinearTimeMMD;
%rename(BTestMMD) CBTestMMD;
%rename(QuadraticTimeMMD) CQuadraticTimeMMD;
%rename(MultiKernelQuadraticTimeMMD) CMultiKernelQuadraticTimeMMD;
%rename(KernelSelectionStrategy) CKernelSelectionStrategy;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/statistical_testing/TestEnums.h>
%include <shogun/statistical_testing/HypothesisTest.h>
%include <shogun/statistical_testing/OneDistributionTest.h>
%include <shogun/statistical_testing/TwoDistributionTest.h>
%include <shogun/statistical_testing/IndependenceTest.h>
%include <shogun/statistical_testing/TwoSampleTest.h>
%include <shogun/statistical_testing/MMD.h>
%include <shogun/statistical_testing/StreamingMMD.h>
%include <shogun/statistical_testing/LinearTimeMMD.h>
%include <shogun/statistical_testing/BTestMMD.h>
%include <shogun/statistical_testing/QuadraticTimeMMD.h>
%include <shogun/statistical_testing/MultiKernelQuadraticTimeMMD.h>
%include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
