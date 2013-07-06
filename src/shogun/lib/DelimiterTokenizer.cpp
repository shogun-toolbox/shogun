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

CDelimiterTokenizer::CDelimiterTokenizer() : delimiters(256)
{
	last_idx = 0;
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
	SGVector<bool>::fill_vector(delimiters, 256, 0);
}

void CDelimiterTokenizer::set_text(SGVector<char> txt)
{
	last_idx = 0;
	CTokenizer::set_text(txt);
}

const char* CDelimiterTokenizer::get_name() const
{
    return "WhiteSpaceTokenizer";
}

bool CDelimiterTokenizer::has_next()
{
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

	if (delimiters[(uint8_t) text[start]]==0)
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
	return t;
}
}
