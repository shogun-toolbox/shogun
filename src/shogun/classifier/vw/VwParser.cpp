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
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <shogun/classifier/vw/VwParser.h>
#include <shogun/classifier/vw/cache/VwNativeCacheWriter.h>

using namespace shogun;

CVwParser::CVwParser()
	: CSGObject()
{
	env = new CVwEnvironment();
	hasher = CHash::MurmurHashString;
	write_cache = false;
	cache_writer = NULL;
}

CVwParser::CVwParser(CVwEnvironment* env_to_use)
	: CSGObject()
{
	ASSERT(env_to_use)

	env = env_to_use;
	hasher = CHash::MurmurHashString;
	write_cache = false;
	cache_writer = NULL;
	SG_REF(env);
}

CVwParser::~CVwParser()
{
	SG_UNREF(env);
	SG_UNREF(cache_writer);
}

int32_t CVwParser::read_features(CIOBuffer* buf, VwExample*& ae)
{
	char *line=NULL;
	int32_t num_chars = buf->read_line(line);
	if (num_chars == 0)
		return num_chars;

	/* Mark begin and end of example in the buffer */
	substring example_string = {line, line + num_chars};

	/* Channels containing separate namespaces/label information*/
	channels.erase();

	/* Split at '|' character */
	tokenize('|', example_string, channels);

	/* If first char is not '|', then the first channel contains label data */
	substring* feature_start = &channels[1];

	if (*line == '|')
		feature_start = &channels[0]; /* Unlabelled data */
	else
	{
		/* First channel has label info */
		substring label_space = channels[0];
		char* tab_location = safe_index(label_space.start, '\t', label_space.end);
		if (tab_location != label_space.end)
			label_space.start = tab_location+1;

		/* Split the label space on spaces */
		tokenize(' ',label_space,words);
		if (words.index() > 0 && words.last().end == label_space.end) //The last field is a tag, so record and strip it off
		{
			substring tag = words.pop();
			ae->tag.push_many(tag.start, tag.end - tag.start);
		}

		ae->ld->label_from_substring(words);
		set_minmax(ae->ld->label);
	}

	vw_size_t mask = env->mask;

	/* Now parse the individual channels, i.e., namespaces */
	for (substring* i = feature_start; i != channels.end; i++)
	{
		substring channel = *i;

		tokenize(' ',channel, words);
		if (words.begin == words.end)
			continue;

		/* Set default scale value for channel */
		float32_t channel_v = 1.;
		vw_size_t channel_hash;

		/* Index by which to refer to the namespace */
		vw_size_t index = 0;
		bool new_index = false;
		vw_size_t feature_offset = 0;

		if (channel.start[0] != ' ')
		{
			/* Nonanonymous namespace specified */
			feature_offset++;
			feature_value(words[0], name, channel_v);

			if (name.index() > 0)
			{
				index = (unsigned char)(*name[0].start);
				if (ae->atomics[index].begin == ae->atomics[index].end)
				{
					ae->sum_feat_sq[index] = 0;
					new_index = true;
				}
			}
			channel_hash = hasher(name[0], hash_base);
		}
		else
		{
			/* Use default namespace with index below */
			index = (unsigned char)' ';
			if (ae->atomics[index].begin == ae->atomics[index].end)
			{
				ae->sum_feat_sq[index] = 0;
				new_index = true;
			}
			channel_hash = 0;
		}

		for (substring* j = words.begin+feature_offset; j != words.end; j++)
		{
			/* Get individual features and multiply by scale value */
			float32_t v = 0.0;
			feature_value(*j, name, v);
			v *= channel_v;

			/* Hash feature */
			vw_size_t word_hash = (hasher(name[0], channel_hash)) & mask;
			VwFeature f = {v,word_hash};
			ae->sum_feat_sq[index] += v*v;
			ae->atomics[index].push(f);
		}

		/* Add index to list of indices if required */
		if (new_index && ae->atomics[index].begin != ae->atomics[index].end)
			ae->indices.push(index);

	}

	if (write_cache)
		cache_writer->cache_example(ae);

	return num_chars;
}

int32_t CVwParser::read_svmlight_features(CIOBuffer* buf, VwExample*& ae)
{
	char *line=NULL;
	int32_t num_chars = buf->read_line(line);
	if (num_chars == 0)
		return num_chars;

	/* Mark begin and end of example in the buffer */
	substring example_string = {line, line + num_chars};

	vw_size_t mask = env->mask;
	tokenize(' ', example_string, words);

	ae->ld->label = SGIO::float_of_substring(words[0]);
	ae->ld->weight = 1.;
	ae->ld->initial = 0.;
	set_minmax(ae->ld->label);

	substring* feature_start = &words[1];

	vw_size_t index = (unsigned char)' ';	// Any default namespace is ok
	vw_size_t channel_hash = 0;
	ae->sum_feat_sq[index] = 0;
	ae->indices.push(index);
	/* Now parse the individual features */
	for (substring* i = feature_start; i != words.end; i++)
	{
		float32_t v;
		feature_value(*i, name, v);

		vw_size_t word_hash = (hasher(name[0], channel_hash)) & mask;
		VwFeature f = {v,word_hash};
		ae->sum_feat_sq[index] += v*v;
		ae->atomics[index].push(f);
	}

	if (write_cache)
		cache_writer->cache_example(ae);

	return num_chars;
}

int32_t CVwParser::read_dense_features(CIOBuffer* buf, VwExample*& ae)
{
	char *line=NULL;
	int32_t num_chars = buf->read_line(line);
	if (num_chars == 0)
		return num_chars;

	// Mark begin and end of example in the buffer
	substring example_string = {line, line + num_chars};

	vw_size_t mask = env->mask;
	tokenize(' ', example_string, words);

	ae->ld->label = SGIO::float_of_substring(words[0]);
	ae->ld->weight = 1.;
	ae->ld->initial = 0.;
	set_minmax(ae->ld->label);

	substring* feature_start = &words[1];

	vw_size_t index = (unsigned char)' ';

	ae->sum_feat_sq[index] = 0;
	ae->indices.push(index);
	// Now parse individual features
	int32_t j=0;
	for (substring* i = feature_start; i != words.end; i++)
	{
		float32_t v = SGIO::float_of_substring(*i);
		vw_size_t word_hash = j & mask;
		VwFeature f = {v,word_hash};
		ae->sum_feat_sq[index] += v*v;
		ae->atomics[index].push(f);
		j++;
	}

	if (write_cache)
		cache_writer->cache_example(ae);

	return num_chars;
}

void CVwParser::init_cache(char * fname, EVwCacheType type)
{
	char* file_name = fname;
	char default_cache_name[] = "vw_cache.dat.cache";

	if (!fname)
		file_name = default_cache_name;

	write_cache = true;
	cache_type = type;

	switch (type)
	{
	case C_NATIVE:
		cache_writer = new CVwNativeCacheWriter(file_name, env);
		return;
	case C_PROTOBUF:
		SG_ERROR("Protocol buffers cache support is not implemented yet.\n")
	}

	SG_ERROR("Unexpected cache type specified!\n")
}

void CVwParser::feature_value(substring &s, v_array<substring>& feat_name, float32_t &v)
{
	// Get the value of the feature in the substring
	tokenize(':', s, feat_name);

	switch (feat_name.index())
	{
	// If feature value is not specified, assume 1.0
	case 0:
	case 1:
		v = 1.;
		break;
	case 2:
		v = SGIO::float_of_substring(feat_name[1]);
		if (CMath::is_nan(v))
			SG_SERROR("error NaN value for feature %s! Terminating!\n",
				  SGIO::c_string_of_substring(feat_name[0]));
		break;
	default:
		SG_SERROR("Examples with a weird name, i.e., '%s'\n",
			  SGIO::c_string_of_substring(s));
	}
}

void CVwParser::tokenize(char delim, substring s, v_array<substring>& ret)
{
	ret.erase();
	char *last = s.start;
	for (; s.start != s.end; s.start++)
	{
		if (*s.start == delim)
		{
			if (s.start != last)
			{
				substring temp = {last,s.start};
				ret.push(temp);
			}
			last = s.start+1;
		}
	}
	if (s.start != last)
	{
		substring final = {last, s.start};
		ret.push(final);
	}
}
