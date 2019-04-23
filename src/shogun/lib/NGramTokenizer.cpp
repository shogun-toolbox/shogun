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

NGramTokenizer::NGramTokenizer(int32_t ns) : Tokenizer()
{
	n = ns;
	last_idx = 0;
	init();
}

NGramTokenizer::NGramTokenizer(const NGramTokenizer& orig)
: Tokenizer(orig)
{
	Tokenizer::set_text(orig.text);
	n = orig.n;
	init();
}

void NGramTokenizer::init()
{
	SG_ADD(&n, "n", "Size of n-grams");
	SG_ADD(&last_idx, "last_idx", "Index of last token");
}

void NGramTokenizer::set_text(SGVector<char> txt)
{
	last_idx = 0;
	Tokenizer::set_text(txt);
}

const char* NGramTokenizer::get_name() const
{
    return "NGramTokenizer";
}

bool NGramTokenizer::has_next()
{
	return last_idx<=text.size()-n;
}

index_t NGramTokenizer::next_token_idx(index_t& start)
{
	start = last_idx++;
	return start + n;
}

NGramTokenizer* NGramTokenizer::get_copy()
{
	NGramTokenizer* t = new NGramTokenizer(n);
	return t;
}
}
