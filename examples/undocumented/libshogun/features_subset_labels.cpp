/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

const int32_t num_labels=10;
const int32_t num_classes=3;

void test()
{
	const int32_t num_subset_idx=CMath::random(1, num_labels);

	/* create labels */
	CMulticlassLabels* labels=new CMulticlassLabels(num_labels);
	for (index_t i=0; i<num_labels; ++i)
		labels->set_label(i, i%num_classes);

	SG_REF(labels);

	/* print labels */
	SGVector<float64_t> labels_data=labels->get_labels();
	SGVector<float64_t>::display_vector(labels_data.vector, labels_data.vlen, "labels");

	/* create subset indices */
	SGVector<index_t> subset_idx(num_subset_idx);
	subset_idx.range_fill();
	CMath::permute(subset_idx);

	/* print subset indices */
	SGVector<index_t>::display_vector(subset_idx.vector, subset_idx.vlen, "subset indices");

	/* apply subset to features */
	SG_SPRINT("\n\n-------------------\n"
			"applying subset to features\n"
			"-------------------\n");
	labels->add_subset(subset_idx);

	/* do some stuff do check and output */
	ASSERT(labels->get_num_labels()==num_subset_idx);
	SG_SPRINT("labels->get_num_labels(): %d\n", labels->get_num_labels());

	for (index_t i=0; i<labels->get_num_labels(); ++i)
	{
		float64_t label=labels->get_label(i);
		SG_SPRINT("label %f:\n", label);
		ASSERT(label==labels_data.vector[subset_idx.vector[i]]);
	}

	/* remove features subset */SG_SPRINT("\n\n-------------------\n"
			"removing subset from features\n"
			"-------------------\n");
	labels->remove_all_subsets();

	ASSERT(labels->get_num_labels()==num_labels);
	SG_SPRINT("labels->get_num_labels(): %d\n", labels->get_num_labels());

	for (index_t i=0; i<labels->get_num_labels(); ++i)
	{
		float64_t label=labels->get_label(i);
		SG_SPRINT("label %f:\n", label);
		ASSERT(label==labels_data.vector[i]);
	}
	SG_UNREF(labels);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test();

	exit_shogun();

	return 0;
}

