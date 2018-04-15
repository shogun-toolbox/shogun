/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Bjoern Esser
 */

#include <shogun/lib/NGramTokenizer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

CNGramTokenizer::CNGramTokenizer(int32_t ns) : CTokenizer()
{
	n = ns;
	last_idx = 0;
	init();
}

CNGramTokenizer::CNGramTokenizer(const CNGramTokenizer& orig)
: CTokenizer(orig)
{
	CTokenizer::set_text(orig.text);
	n = orig.n;
	init();
}

void CNGramTokenizer::init()
{
	SG_ADD(&n, "n", "Size of n-grams",
		MS_NOT_AVAILABLE);
	SG_ADD(&last_idx, "last_idx", "Index of last token",
		MS_NOT_AVAILABLE);
}

void CNGramTokenizer::set_text(SGVector<char> txt)
{
	last_idx = 0;
	CTokenizer::set_text(txt);
}

const char* CNGramTokenizer::get_name() const
{
    return "NGramTokenizer";
}

bool CNGramTokenizer::has_next()
{
	return last_idx<=text.size()-n;
}

index_t CNGramTokenizer::next_token_idx(index_t& start)
{
	start = last_idx++;
	return start + n;
}

CNGramTokenizer* CNGramTokenizer::get_copy()
{
	CNGramTokenizer* t = new CNGramTokenizer(n);
	return t;
}
}
