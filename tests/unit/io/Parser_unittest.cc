#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/Parser.h>
#include <shogun/io/SGIO.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(ParserTest, tokenization)
{
	int32_t ntokens=5;
	const char* text="	This is  	the ultimate test!	";
	const char* tokens[]={"This", "is", "the", "ultimate", "test!"};
	SGVector<char> cv(const_cast<char* >(text), 30, false);

	CDelimiterTokenizer* tokenizer=new CDelimiterTokenizer();
	tokenizer->init_for_whitespace();
	tokenizer->set_skip_delimiters(true);
	SG_REF(tokenizer);

	CParser* reader=new CParser(cv, tokenizer);

	SGVector<char> token;
	int32_t num_tokens=0;
	while (reader->has_next())
	{
		token=reader->read_string();

		EXPECT_EQ((index_t) strlen(tokens[num_tokens]), token.vlen);
		for (int32_t i=0; i<token.vlen; i++)
		{
			EXPECT_EQ(tokens[num_tokens][i], token[i]);
		}
		num_tokens++;
	}
	EXPECT_EQ(num_tokens, ntokens);

	SG_UNREF(reader);
	SG_UNREF(tokenizer);
}

TEST(ParserTest, tokenization_reals)
{
	int32_t ntokens=5;
	const char* text="1.0, 1.1, 1.2, 1.3, 1.4\n";
	float64_t tokens[]={1.0, 1.1, 1.2, 1.3, 1.4};
	SGVector<char> cv(const_cast<char* >(text), 24, false);

	CDelimiterTokenizer* tokenizer=new CDelimiterTokenizer();
	tokenizer->delimiters[' ']=1;
	tokenizer->delimiters[',']=1;
	tokenizer->delimiters['\n']=1;
	tokenizer->set_skip_delimiters(true);
	SG_REF(tokenizer);

	CParser* reader=new CParser(cv, tokenizer);

	SG_SET_LOCALE_C;

	float64_t tmp=0;
	int32_t num_tokens=0;
	while (reader->has_next())
	{
		tmp=reader->read_real();
		EXPECT_EQ(tokens[num_tokens], tmp);
		num_tokens++;
	}
	EXPECT_EQ(num_tokens, ntokens);

	SG_RESET_LOCALE;

	SG_UNREF(reader);
	SG_UNREF(tokenizer);
}
