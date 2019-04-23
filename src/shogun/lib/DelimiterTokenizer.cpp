/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Heiko Strathmann, 
 *          Evgeniy Andreev
 */

#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>

#include <string.h>

namespace shogun
{

DelimiterTokenizer::DelimiterTokenizer(bool skip_delimiters) : delimiters(256)
{
	last_idx = 0;
	skip_consecutive_delimiters = skip_delimiters;
	init();
}

DelimiterTokenizer::DelimiterTokenizer(const DelimiterTokenizer& orig)
{
	Tokenizer::set_text(orig.text);
	delimiters = orig.delimiters;
	init();
}

void DelimiterTokenizer::init()
{
	SG_ADD(&last_idx, "last_idx", "Index of last token");
	SG_ADD(&skip_consecutive_delimiters, "skip_consecutive_delimiters",
		"Whether to skip consecutive delimiters or not");
	SGVector<bool>::fill_vector(delimiters, 256, 0);
}

void DelimiterTokenizer::set_text(SGVector<char> txt)
{
	last_idx = 0;
	Tokenizer::set_text(txt);
}

const char* DelimiterTokenizer::get_name() const
{
    return "DelimiterTokenizer";
}

bool DelimiterTokenizer::has_next()
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

void DelimiterTokenizer::init_for_whitespace()
{
	clear_delimiters();
	delimiters[' '] = 1;
	delimiters['\t'] = 1;
}

void DelimiterTokenizer::clear_delimiters()
{
	memset(delimiters, 0, sizeof (delimiters));
}

index_t DelimiterTokenizer::next_token_idx(index_t& start)
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

DelimiterTokenizer* DelimiterTokenizer::get_copy()
{
	DelimiterTokenizer* t = new DelimiterTokenizer();
	t->delimiters = delimiters;
	t->skip_consecutive_delimiters = skip_consecutive_delimiters;
	return t;
}

void DelimiterTokenizer::set_skip_delimiters(bool skip_delimiters)
{
	skip_consecutive_delimiters = skip_delimiters;
}

bool DelimiterTokenizer::get_skip_delimiters() const
{
	return skip_consecutive_delimiters;
}
}
