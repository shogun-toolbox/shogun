/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Thoralf Klein, Sahil Chaddha, Soeren Sonnenburg, 
 *          Bjoern Esser
 */

#include <gtest/gtest.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/LOOCrossValidationSplitting.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/TimeSeriesSplitting.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#include <random>

using namespace shogun;

TEST(SplittingStrategy,standard)
{
	int32_t seed = 12;
	index_t fold_sizes;
	index_t num_labels;
	index_t num_subsets;
	index_t runs=100;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	UniformRealDistribution<float64_t> uniform_real_dist;
	while (runs-->0)
	{
		fold_sizes=0;
		num_labels=uniform_int_dist(prng, {10, 150});
		num_subsets=uniform_int_dist(prng, {1, 5});
		index_t desired_size=Math::round(
				(float64_t)num_labels/(float64_t)num_subsets);

		/* build labels */
		auto labels=std::make_shared<RegressionLabels>(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, uniform_real_dist(prng, {-10.0, 10.0}));

		/* build splitting strategy */
		CrossValidationSplitting* splitting=
				new CrossValidationSplitting(labels, num_subsets);

		splitting->build_subsets();

		SGVector<index_t> total(num_labels);
		SGVector<index_t>::fill_vector(total.vector, total.vlen,(index_t)-1);

		for (index_t i=0; i<num_subsets; ++i)
		{
			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);

			for(index_t j=0;j<subset.vlen;++j)
			{
				/*check if fold indices are disjoint*/
				SGVector<index_t> temp=total.find((index_t)subset.vector[j]);
				EXPECT_EQ(temp.vlen,0);

				total.vector[j+fold_sizes]=subset.vector[j];
			}

			fold_sizes+=subset.vlen;

			EXPECT_LE(Math::abs(subset.vlen-desired_size),1);
			EXPECT_EQ(subset.vlen+inverse.vlen,num_labels);
		}

		EXPECT_EQ(fold_sizes,num_labels);

		index_t flag=0;
		/*check if indices in all folds cover available indices*/
		for (index_t i=0;i<num_labels;++i)
		{
			SGVector<index_t> temp=total.find((index_t)i);
			if(temp.vlen == 0)
				flag = 1;
		}

		EXPECT_EQ(flag,0);

		/* clean up */
		
	}

}

TEST(SplittingStrategy,stratified_subsets_disjoint_cover)
{
	int32_t seed = 12;
	index_t num_labels, num_classes, num_subsets, fold_sizes;
	index_t runs=50;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	while (runs-->0)
	{
		fold_sizes=0;
		num_labels=uniform_int_dist(prng, {11, 100});
		num_classes=uniform_int_dist(prng, {2, 10});
		num_subsets=uniform_int_dist(prng, {1, 10});

		/* build labels */
		auto labels=std::make_shared<MulticlassLabels>(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, prng()%num_classes);

		SGVector<float64_t> classes=labels->get_unique_labels();

		/*No. of labels belonging to one class*/
		SGVector<index_t> class_labels(num_classes);
		SGVector<index_t>::fill_vector(class_labels.vector, class_labels.vlen, 0);

		/*check total no. of class labels*/
		for (index_t i=0; i<num_classes; ++i)
		{
			for(index_t j=0; j<num_labels; ++j)
			{
			       if ((int32_t)labels->get_label(j)==i)
				       ++class_labels.vector[i];
			}
		}


		/* build splitting strategy */
		StratifiedCrossValidationSplitting* splitting=
				new StratifiedCrossValidationSplitting(labels, num_subsets);

		splitting->build_subsets();

		SGVector<index_t> total(num_labels);
		SGVector<index_t>::fill_vector(total.vector, total.vlen,(index_t)-1);

		for (index_t i=0; i<num_subsets; ++i)
		{
			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);

			for(index_t j=0;j<subset.vlen;++j)
			{
				/*check if fold indices are disjoint*/
				SGVector<index_t> temp=total.find((index_t)subset.vector[j]);
				EXPECT_EQ(temp.vlen,0);

				total.vector[j+fold_sizes]=subset.vector[j];
			}

			EXPECT_EQ(subset.vlen+inverse.vlen, num_labels);
			fold_sizes+=subset.vlen;
		}

		EXPECT_EQ(fold_sizes, num_labels);

		index_t flag=0;
		/*check if indices in all folds cover available indices*/
		for (index_t i=0;i<num_labels;++i)
		{
			SGVector<index_t> temp=total.find((index_t)i);
			if(temp.vlen == 0)
				flag = 1;
		}

		EXPECT_EQ(flag,0);

		/* clean up */
		
	}
}

TEST(SplittingStrategy,stratified_subset_label_ratio)
{
	int32_t seed = 12;
	index_t num_labels, num_classes, num_subsets;
	index_t runs=50;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	while (runs-->0)
	{
		num_labels=uniform_int_dist(prng, {11, 100});
		num_classes=uniform_int_dist(prng, {2, 10});
		num_subsets=uniform_int_dist(prng, {1, 10});

		/* build labels */
		auto labels=std::make_shared<MulticlassLabels>(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, prng()%num_classes);

		/*No. of labels belonging to one class*/
		SGVector<index_t> class_labels(num_classes);
		SGVector<index_t>::fill_vector(class_labels.vector, class_labels.vlen, 0);

		/*check total no. of class labels*/
		for (index_t i=0; i<num_classes; ++i)
		{
			for(index_t j=0; j<num_labels; ++j)
			{
			       if ((int32_t)labels->get_label(j)==i)
				       ++class_labels.vector[i];
			}
		}


		/* build splitting strategy */
		StratifiedCrossValidationSplitting* splitting=
				new StratifiedCrossValidationSplitting(labels, num_subsets);

		splitting->build_subsets();

		/* check whether number of labels in every subset is nearly equal */
		for (index_t i=0; i<num_classes; ++i)
		{
			/* count number of elements for this class */
			SGVector<index_t> temp=splitting->generate_subset_indices(0);
			int32_t count=0;
			int32_t total_count=0;
			for (index_t j=0; j<temp.vlen; ++j)
			{
				if ((int32_t)labels->get_label(temp.vector[j])==i)
					++count;
			}

			/* check all subsets for same ratio */
			for (index_t j=0; j<num_subsets; ++j)
			{
				SGVector<index_t> subset=splitting->generate_subset_indices(j);
				int32_t temp_count=0;
				for (index_t k=0; k<subset.vlen; ++k)
				{
					if ((int32_t)labels->get_label(subset.vector[k])==i)
						++temp_count;
				}

				total_count+=temp_count;
				/* at most one difference */
				EXPECT_LE(Math::abs(temp_count-count),1);
			}
			EXPECT_EQ(total_count,class_labels.vector[i]);
		}
		/* clean up */
		
	}
}


TEST(SplittingStrategy,LOO)
{
	int32_t seed = 12;
	index_t num_labels, fold_sizes;
	index_t runs=10;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	UniformRealDistribution<float64_t> uniform_real_dist;
	while (runs-->0)
	{
		fold_sizes=0;
		num_labels=uniform_int_dist(prng, {10, 50});

		/* build labels */
		auto labels=std::make_shared<RegressionLabels>(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, uniform_real_dist(prng, {-10.0, 10.0}));

		/* build Leave one out splitting strategy */
		LOOCrossValidationSplitting* splitting=
				new LOOCrossValidationSplitting(labels);

		splitting->build_subsets();

		SGVector<index_t> total(num_labels);
		SGVector<index_t>::fill_vector(total.vector, total.vlen,(index_t)-1);

		for (index_t i=0; i<num_labels; ++i)
		{
			SGVector<index_t> subset=splitting->generate_subset_indices(i);
			SGVector<index_t> inverse=splitting->generate_subset_inverse(i);

			for(index_t j=0;j<subset.vlen;++j)
			{
				/*check if fold indices are disjoint*/
				SGVector<index_t> temp=total.find((index_t)subset.vector[j]);
				EXPECT_EQ(temp.vlen,0);

				total.vector[j+fold_sizes]=subset.vector[j];
			}

			EXPECT_EQ(subset.vlen+inverse.vlen, num_labels);
			fold_sizes+=subset.vlen;
		}

		EXPECT_EQ(fold_sizes, num_labels);

		index_t flag=0;
		/*check if indices in all folds cover available indices*/
		for (index_t i=0;i<num_labels;++i)
		{
			SGVector<index_t> temp=total.find((index_t)i);
			if(temp.vlen == 0)
				flag = 1;
		}

		EXPECT_EQ(flag,0);

		/* clean up */
		
	}
}

TEST(SplittingStrategy, timeseries_subset_linear_splits)
{
	int32_t seed = 20;
	index_t num_labels, num_subsets, min_subset_size, base_size;
	index_t runs = 10;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	UniformRealDistribution<float64_t> uniform_real_dist;
	while (runs-- > 0)
	{
		num_labels = uniform_int_dist(prng, {50, 150});
		num_subsets = uniform_int_dist(prng, {1, 5});
		min_subset_size = uniform_int_dist(prng, {1, 6});
		base_size = num_labels / num_subsets;

		auto labels = std::make_shared<RegressionLabels>(num_labels);
		for (index_t i = 0; i < num_labels; ++i)
			labels->set_label(i, uniform_real_dist(prng, {-10.0, 10.0}));

		TimeSeriesSplitting* splitting =
		    new TimeSeriesSplitting(labels, num_subsets);

		splitting->set_min_subset_size(min_subset_size);
		splitting->build_subsets();

		for (index_t i = 0; i < num_subsets; ++i)
		{
			SGVector<index_t> subset = splitting->generate_subset_indices(i);
			SGVector<index_t> inverse = splitting->generate_subset_inverse(i);

			/* Subset size should be atleat min_subset_size */
			EXPECT_GE(subset.vlen, splitting->get_min_subset_size());

			/* check the splitting is linear */
			EXPECT_TRUE(
			    inverse.vlen % base_size == 0 ||
			    inverse.vlen == num_labels - min_subset_size);
		}

		
	}
}

TEST(SplittingStrategy, timeseries_subsets_future_leak)
{
	int32_t seed = 12;
	index_t num_labels, num_subsets, min_subset_size;
	index_t runs = 10;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	UniformRealDistribution<float64_t> uniform_real_dist;
	while (runs-- > 0)
	{
		num_labels = uniform_int_dist(prng, {50, 150});
		num_subsets = uniform_int_dist(prng, {1, 5});
		min_subset_size = uniform_int_dist(prng, {1, 7});

		auto labels = std::make_shared<RegressionLabels>(num_labels);
		for (index_t i = 0; i < num_labels; ++i)
			labels->set_label(i, uniform_real_dist(prng, {-10.0, 10.0}));

		TimeSeriesSplitting* splitting =
		    new TimeSeriesSplitting(labels, num_subsets);

		splitting->set_min_subset_size(min_subset_size);
		splitting->build_subsets();

		for (index_t i = 0; i < num_subsets; ++i)
		{
			SGVector<index_t> subset = splitting->generate_subset_indices(i);
			SGVector<index_t> inverse = splitting->generate_subset_inverse(i);

			/* check future leak into test set */
			for (index_t j = 0; j < inverse.vlen - 1; ++j)
			{
				EXPECT_LT(inverse.vector[j], inverse.vector[j + 1]);
			}

			EXPECT_LT(inverse.vector[inverse.vlen - 1], subset.vector[0]);

			for (index_t j = 0; j < subset.vlen - 1; ++j)
			{
				EXPECT_LT(subset.vector[j], subset.vector[j + 1]);
			}

			EXPECT_EQ(subset.vlen + inverse.vlen, num_labels);
		}

		/* clean up */
		
	}
}
