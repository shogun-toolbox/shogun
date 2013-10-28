/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/base/Parameter.h>
#include <shogun/lib/DelimiterTokenizer.h>

namespace shogun
{

CDelimiterTokenizer::CDelimiterTokenizer(bool skip_delimiters) : delimiters(256)
{
	last_idx = 0;
	skip_consecutive_delimiters = skip_delimiters;
	init();
}

CDelimiterTokenizer::CDelimiterTokenizer(const CDelimiterTokenizer& orig)
{
	CTokenizer::set_text(orig.text);
	delimiters = orig.delimiters;
	init();
}

void CDelimiterTokenizer::init()
{
	SG_ADD(&last_idx, "last_idx", "Index of last token",
		MS_NOT_AVAILABLE);
	SG_ADD(&skip_consecutive_delimiters, "skip_consecutive_delimiters",
		"Whether to skip consecutive delimiters or not", MS_NOT_AVAILABLE);
	SGVector<bool>::fill_vector(delimiters, 256, 0);
}

void CDelimiterTokenizer::set_text(SGVector<char> txt)
{
	last_idx = 0;
	CTokenizer::set_text(txt);
}

const char* CDelimiterTokenizer::get_name() const
{
    return "DelimiterTokenizer";
}

bool CDelimiterTokenizer::has_next()
{
	if (skip_consecutive_delimiters)
	{
		for (index_t i=last_idx; i<text.size(); i++)
		{
			if (! delimiters[(uint8_t) text[i]])
				return true;
		}
		return false;
	}
	else
		return last_idx<text.size();
}

void CDelimiterTokenizer::init_for_whitespace()
{
	clear_delimiters();
	delimiters[' '] = 1;
	delimiters['\t'] = 1;
}

void CDelimiterTokenizer::clear_delimiters()
{
	memset(delimiters, 0, sizeof (delimiters));
}

index_t CDelimiterTokenizer::next_token_idx(index_t& start)
{
	start = last_idx;

	if (skip_consecutive_delimiters)
	{
		while(delimiters[(uint8_t) text[start]])
			start++;
	}

	if (! delimiters[(uint8_t) text[start]])
	{
		for (last_idx=start+1; last_idx<text.size(); last_idx++)
		{
			if (delimiters[(uint8_t) text[last_idx]])
				break;
		}
	}

	return last_idx++;
}

CDelimiterTokenizer* CDelimiterTokenizer::get_copy()
{
	CDelimiterTokenizer* t = new CDelimiterTokenizer();
	t->delimiters = delimiters;
	t->skip_consecutive_delimiters = skip_consecutive_delimiters;
	return t;
}

void CDelimiterTokenizer::set_skip_delimiters(bool skip_delimiters)
{
	skip_consecutive_delimiters = skip_delimiters;
}

bool CDelimiterTokenizer::get_skip_delimiters() const
{
	return skip_consecutive_delimiters;
}
}
