#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(DelimiterTokenizerTest, tokenization)
{
	const char* text = "	This is  	the ultimate test!	";
	const char* tokens[] = {"This", "is", "the", "ultimate", "test!"};
	SGVector<char> cv(const_cast<char* >(text), 30, false);

	CDelimiterTokenizer* tokenizer = new CDelimiterTokenizer();
	tokenizer->init_for_whitespace();
	tokenizer->set_text(cv);

	index_t token_start = 0;
	index_t token_in_tokens = 0;
	while (tokenizer->has_next())
	{
		index_t token_end = tokenizer->next_token_idx(token_start);
		if (token_end==token_start)
			continue;

		char token[token_end-token_start+1];
		for (index_t i=token_start; i<token_end; i++)
		{
			token[i-token_start] = text[i];
		}
		token[token_end-token_start] = '\0';
		ASSERT_STREQ(token, tokens[token_in_tokens++]);
	}

	ASSERT_EQ(token_in_tokens, 5);

	tokenizer->set_skip_delimiters(true);
	tokenizer->set_text(cv);
	token_in_tokens = 0;
	while (tokenizer->has_next())
	{
		index_t token_end = tokenizer->next_token_idx(token_start);
		char token[token_end-token_start+1];
		for (index_t i=token_start; i<token_end; i++)
		{
			token[i-token_start] = text[i];
		}
		token[token_end-token_start] = '\0';
		ASSERT_STREQ(token, tokens[token_in_tokens++]);
	}

	ASSERT_EQ(token_in_tokens, 5);
	SG_UNREF(tokenizer);
}
