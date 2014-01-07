/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#include <base/init.h>
#include <evaluation/CrossValidationSplitting.h>
#include <labels/RegressionLabels.h>
#include <evaluation/StratifiedCrossValidationSplitting.h>
#include <labels/MulticlassLabels.h>
#include <evaluation/LOOCrossValidationSplitting.h>
#include <gtest/gtest.h>


using namespace shogun;

TEST(SplittingStrategy,standard)
{	
	index_t fold_sizes;
	index_t num_labels;
	index_t num_subsets;
	index_t runs=100;

	while (runs-->0)
	{	
		fold_sizes=0;
		num_labels=CMath::random(10, 150);
		num_subsets=CMath::random(1, 5);
		index_t desired_size=CMath::round(
				(float64_t)num_labels/(float64_t)num_subsets);

		/* build labels */
		CRegressionLabels* labels=new CRegressionLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, CMath::random(-10.0, 10.0));
		
		/* build splitting strategy */
		CCrossValidationSplitting* splitting=
				new CCrossValidationSplitting(labels, num_subsets);

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

			EXPECT_LE(CMath::abs(subset.vlen-desired_size),1);
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
		SG_UNREF(splitting);
	}

}

TEST(SplittingStrategy,stratified_subsets_disjoint_cover)
{
	index_t num_labels, num_classes, num_subsets, fold_sizes;
	index_t runs=50;

	while (runs-->0)
	{	
		fold_sizes=0;
		num_labels=CMath::random(11, 100);
		num_classes=CMath::random(2, 10);
		num_subsets=CMath::random(1, 10);

		/* build labels */
		CMulticlassLabels* labels=new CMulticlassLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, CMath::random()%num_classes);

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
		CStratifiedCrossValidationSplitting* splitting=
				new CStratifiedCrossValidationSplitting(labels, num_subsets);

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
		SG_UNREF(splitting);
	}
}

TEST(SplittingStrategy,stratified_subset_label_ratio)
{
	index_t num_labels, num_classes, num_subsets;
	index_t runs=50;

	while (runs-->0)
	{	
		num_labels=CMath::random(11, 100);
		num_classes=CMath::random(2, 10);
		num_subsets=CMath::random(1, 10);

		/* build labels */
		CMulticlassLabels* labels=new CMulticlassLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, CMath::random()%num_classes);

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
		CStratifiedCrossValidationSplitting* splitting=
				new CStratifiedCrossValidationSplitting(labels, num_subsets);

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
				EXPECT_LE(CMath::abs(temp_count-count),1);
			}
			EXPECT_EQ(total_count,class_labels.vector[i]);
		}
		/* clean up */
		SG_UNREF(splitting);
	}
}


TEST(SplittingStrategy,LOO)
{
	index_t num_labels, fold_sizes;
	index_t runs=10;

	while (runs-->0)
	{
		fold_sizes=0;
		num_labels=CMath::random(10, 50);

		/* build labels */
		CRegressionLabels* labels=new CRegressionLabels(num_labels);
		for (index_t i=0; i<num_labels; ++i)
			labels->set_label(i, CMath::random(-10.0, 10.0));
		
		/* build Leave one out splitting strategy */
		CLOOCrossValidationSplitting* splitting=
				new CLOOCrossValidationSplitting(labels);
		
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
		SG_UNREF(splitting);
	}
}
