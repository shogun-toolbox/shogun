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

#include <shogun/classifier/vw/VwRegressor.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/io/IOBuffer.h>

using namespace shogun;

CVwRegressor::CVwRegressor()
	: CSGObject()
{
	weight_vectors = NULL;
	loss = new CSquaredLoss();
	init(NULL);
}

CVwRegressor::CVwRegressor(CVwEnvironment* env_to_use)
	: CSGObject()
{
	weight_vectors = NULL;
	loss = new CSquaredLoss();
	init(env_to_use);
}

CVwRegressor::~CVwRegressor()
{
	// TODO: the number of weight_vectors depends on num_threads
	SG_FREE(weight_vectors[0]);
	SG_FREE(weight_vectors);
	SG_UNREF(loss);
	SG_UNREF(env);
}

void CVwRegressor::init(CVwEnvironment* env_to_use)
{
	if (!env_to_use)
		env_to_use = new CVwEnvironment();

	env = env_to_use;
	SG_REF(env);

	// For each feature, there should be 'stride' number of
	// elements in the weight vector
	vw_size_t length = ((vw_size_t) 1) << env->num_bits;
	env->thread_mask = (env->stride * (length >> env->thread_bits)) - 1;

	// Only one learning thread for now
	vw_size_t num_threads = 1;
	weight_vectors = SG_MALLOC(float32_t*, num_threads);

	for (vw_size_t i = 0; i < num_threads; i++)
	{
		weight_vectors[i] = SG_CALLOC(float32_t, env->stride * length / num_threads);

		if (env->random_weights)
		{
			for (vw_size_t j = 0; j < length/num_threads; j++)
				weight_vectors[i][j] = CMath::random(-0.5, 0.5);
		}

		if (env->initial_weight != 0.)
			for (vw_size_t j = 0; j < env->stride*length/num_threads; j+=env->stride)
				weight_vectors[i][j] = env->initial_weight;

		if (env->adaptive)
			for (vw_size_t j = 1; j < env->stride*length/num_threads; j+=env->stride)
				weight_vectors[i][j] = 1;
	}
}

void CVwRegressor::dump_regressor(char* reg_name, bool as_text)
{
	CIOBuffer io_temp;
	int32_t f = io_temp.open_file(reg_name,'w');

	if (f < 0)
		SG_SERROR("Can't open: %s for writing! Exiting.\n", reg_name)

	const char* vw_version = env->vw_version;
	vw_size_t v_length = env->v_length;

	if (!as_text)
	{
		// Write version info
		io_temp.write_file((char*)&v_length, sizeof(v_length));
		io_temp.write_file(vw_version,v_length);

		// Write max and min labels
		io_temp.write_file((char*)&env->min_label, sizeof(env->min_label));
		io_temp.write_file((char*)&env->max_label, sizeof(env->max_label));

		// Write weight vector bits information
		io_temp.write_file((char *)&env->num_bits, sizeof(env->num_bits));
		io_temp.write_file((char *)&env->thread_bits, sizeof(env->thread_bits));

		// For paired namespaces forming quadratic features
		int32_t len = env->pairs.get_num_elements();
		io_temp.write_file((char *)&len, sizeof(len));

		for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
			io_temp.write_file(env->pairs.get_element(k), 2);

		// ngram and skips information
		io_temp.write_file((char*)&env->ngram, sizeof(env->ngram));
		io_temp.write_file((char*)&env->skips, sizeof(env->skips));
	}
	else
	{
		// Write as human readable form
		char buff[512];
		int32_t len;

		len = sprintf(buff, "Version %s\n", vw_version);
		io_temp.write_file(buff, len);
		len = sprintf(buff, "Min label:%f max label:%f\n", env->min_label, env->max_label);
		io_temp.write_file(buff, len);
		len = sprintf(buff, "bits:%d thread_bits:%d\n", (int32_t)env->num_bits, (int32_t)env->thread_bits);
		io_temp.write_file(buff, len);

		if (env->pairs.get_num_elements() > 0)
		{
			len = sprintf(buff, "\n");
			io_temp.write_file(buff, len);
		}

		len = sprintf(buff, "ngram:%d skips:%d\nindex:weight pairs:\n", (int32_t)env->ngram, (int32_t)env->skips);
		io_temp.write_file(buff, len);
	}

	uint32_t length = 1 << env->num_bits;
	vw_size_t num_threads = env->num_threads();
	vw_size_t stride = env->stride;

	// Write individual weights
	for(uint32_t i = 0; i < length; i++)
	{
		float32_t v;
		v = weight_vectors[i%num_threads][stride*(i/num_threads)];
		if (v != 0.)
		{
			if (!as_text)
			{
				io_temp.write_file((char *)&i, sizeof (i));
				io_temp.write_file((char *)&v, sizeof (v));
			}
			else
			{
				char buff[512];
				int32_t len = sprintf(buff, "%d:%f\n", i, v);
				io_temp.write_file(buff, len);
			}
		}
	}

	io_temp.close_file();
}

void CVwRegressor::load_regressor(char* file)
{
	CIOBuffer source;
	int32_t fd = source.open_file(file, 'r');

	if (fd < 0)
		SG_SERROR("Unable to open file for loading regressor!\n")

	// Read version info
	vw_size_t v_length;
	source.read_file((char*)&v_length, sizeof(v_length));
	char* t = SG_MALLOC(char, v_length);
	source.read_file(t,v_length);
	if (strcmp(t,env->vw_version) != 0)
	{
		SG_FREE(t);
		SG_SERROR("Regressor source has an incompatible VW version!\n")
	}
	SG_FREE(t);

	// Read min and max label
	source.read_file((char*)&env->min_label, sizeof(env->min_label));
	source.read_file((char*)&env->max_label, sizeof(env->max_label));

	// Read num_bits, multiple sources are not supported
	vw_size_t local_num_bits;
	source.read_file((char *)&local_num_bits, sizeof(local_num_bits));

	if ((vw_size_t) env->num_bits != local_num_bits)
		SG_SERROR("Wrong number of bits in regressor source!\n")

	env->num_bits = local_num_bits;

	vw_size_t local_thread_bits;
	source.read_file((char*)&local_thread_bits, sizeof(local_thread_bits));

	env->thread_bits = local_thread_bits;

	int32_t len;
	source.read_file((char *)&len, sizeof(len));

	// Read paired namespace information
	DynArray<char*> local_pairs;
	for (; len > 0; len--)
	{
		char pair[3];
		source.read_file(pair, sizeof(char)*2);
		pair[2]='\0';
		local_pairs.push_back(pair);
	}

	env->pairs = local_pairs;

	// Initialize the weight vector
	if (weight_vectors)
		SG_FREE(weight_vectors);
	init(env);

	vw_size_t local_ngram;
	source.read_file((char*)&local_ngram, sizeof(local_ngram));
	vw_size_t local_skips;
	source.read_file((char*)&local_skips, sizeof(local_skips));

	env->ngram = local_ngram;
	env->skips = local_skips;

	// Read individual weights
	vw_size_t stride = env->stride;
	while (true)
	{
		uint32_t hash;
		ssize_t hash_bytes = source.read_file((char *)&hash, sizeof(hash));
		if (hash_bytes <= 0)
			break;

		float32_t w = 0.;
		ssize_t weight_bytes = source.read_file((char *)&w, sizeof(float32_t));
		if (weight_bytes <= 0)
			break;

		vw_size_t num_threads = env->num_threads();

		weight_vectors[hash % num_threads][(hash*stride)/num_threads]
			= weight_vectors[hash % num_threads][(hash*stride)/num_threads] + w;
	}
	source.close_file();
}
