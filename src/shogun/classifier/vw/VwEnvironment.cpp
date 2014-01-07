/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <classifier/vw/VwEnvironment.h>

using namespace shogun;

CVwEnvironment::CVwEnvironment()
	: CSGObject(), vw_version("5.1"), v_length(4)
{
	init();
}

void CVwEnvironment::init()
{
	num_bits = 18;
	thread_bits = 0;
	mask = (1 << num_bits) - 1;
	stride = 1;

	min_label = 0.;
	max_label = 1.;

	eta = 10.;
	eta_decay_rate = 1.;

	adaptive = false;
	exact_adaptive_norm = false;
	l1_regularization = 0.;

	random_weights = false;
	initial_weight = 0.;

	update_sum = 0.;

	t = 1.;
	initial_t = 1.;
	power_t = 0.5;

	example_number = 0;
	weighted_examples = 0.;
	weighted_unlabeled_examples = 0.;
	weighted_labels = 0.;
	total_features = 0;
	sum_loss = 0.;
	passes_complete = 0;
	num_passes = 1;

	ngram = 0;
	skips = 0;

	ignore_some = false;

	vw_size_t len = ((vw_size_t) 1) << num_bits;
	thread_mask = (stride * (len >> thread_bits)) - 1;
}

void CVwEnvironment::set_stride(vw_size_t new_stride)
{
	stride = new_stride;
	vw_size_t len = ((vw_size_t) 1) << num_bits;
	thread_mask = (stride * (len >> thread_bits)) - 1;
}
