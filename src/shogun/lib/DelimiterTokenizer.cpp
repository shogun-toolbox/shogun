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

CDelimiterTokenizer::CDelimiterTokenizer()
{
	last_idx = 0;
	init();
}
CDelimiterTokenizer::CDelimiterTokenizer(const CDelimiterTokenizer& orig)
{
	CTokenizer::set_text(orig.text);
	init();
}

void CDelimiterTokenizer::init()
{
	SG_ADD(&last_idx, "last_idx", "Index of last token",
		MS_NOT_AVAILABLE);
	memset(delimiters, 0, sizeof (delimiters));
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
	for (index_t i=last_idx; i<text.size(); i++)
	{
		if (!delimiters[(uint8_t) text[i]])
			return true;
	}
	return false; 
}

void CDelimiterTokenizer::init_for_whitespace()
{
	memset(delimiters, 0, sizeof (delimiters));
	delimiters[' '] = 1;
	delimiters['\t'] = 1;
}

index_t CDelimiterTokenizer::next_token_idx(index_t& start)
{
	while (delimiters[(uint8_t) text[last_idx]])
	{
		last_idx++;
	}
	start = last_idx;
 
	for (last_idx=start+1; last_idx<text.size(); last_idx++)
	{
		if (delimiters[(uint8_t) text[last_idx]])
			break;
	}

	return last_idx;
}
}
