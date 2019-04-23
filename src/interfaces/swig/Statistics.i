/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

%shared_ptr(shogun::HypothesisTest)
%shared_ptr(shogun::OneDistributionTest)
%shared_ptr(shogun::TwoDistributionTest)
%shared_ptr(shogun::IndependenceTest)
%shared_ptr(shogun::TwoSampleTest)
%shared_ptr(shogun::MMD)
%shared_ptr(shogun::StreamingMMD)
%shared_ptr(shogun::LinearTimeMMD)
%shared_ptr(shogun::BTestMMD)
%shared_ptr(shogun::QuadraticTimeMMD)
%shared_ptr(shogun::MultiKernelQuadraticTimeMMD)
%shared_ptr(shogun::KernelSelectionStrategy)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/statistical_testing/TestEnums.h>
%include <shogun/statistical_testing/HypothesisTest.h>
%include <shogun/statistical_testing/OneDistributionTest.h>
%include <shogun/statistical_testing/TwoDistributionTest.h>
%include <shogun/statistical_testing/IndependenceTest.h>
%include <shogun/statistical_testing/TwoSampleTest.h>

/** Instantiate RandomMixin */
%template(SeedableTwoSampleTest) shogun::Seedable<shogun::CTwoSampleTest>;
%template(RandomMixinTwoSampleTest) shogun::RandomMixin<shogun::CTwoSampleTest, std::mt19937_64>;

%include <shogun/statistical_testing/MMD.h>
%include <shogun/statistical_testing/StreamingMMD.h>
%include <shogun/statistical_testing/LinearTimeMMD.h>
%include <shogun/statistical_testing/BTestMMD.h>
%include <shogun/statistical_testing/QuadraticTimeMMD.h>
%include <shogun/statistical_testing/MultiKernelQuadraticTimeMMD.h>
%include <shogun/statistical_testing/kernelselection/KernelSelectionStrategy.h>
