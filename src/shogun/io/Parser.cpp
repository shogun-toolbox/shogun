/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evgeniy Andreev, Soeren Sonnenburg, Thoralf Klein, Bjoern Esser
 */

#include <stdlib.h>
#include <shogun/io/Parser.h>
#include <shogun/lib/Tokenizer.h>

using namespace shogun;

Parser::Parser()
{
	init();
}

Parser::Parser(SGVector<char> text, std::shared_ptr<Tokenizer> tokenizer)
{
	init();

	m_text=text;

	
	m_tokenizer=tokenizer;

	if (m_tokenizer!=NULL)
		m_tokenizer->set_text(m_text);
}

Parser::~Parser()
{
	
}

bool Parser::has_next()
{
	if (m_tokenizer!=NULL)
		return m_tokenizer->has_next();

	return false;
}

void Parser::skip_token()
{
	index_t start=0;
	m_tokenizer->next_token_idx(start);
}

SGVector<char> Parser::read_string()
{
	index_t start=0;
	index_t end=0;

	end=m_tokenizer->next_token_idx(start);

	SGVector<char> result=SGVector<char>(end-start);
	for (index_t i=start; i<end; i++)
	{
		result[i-start]=m_text[i];
	}

	return result;
}

SGVector<char> Parser::read_cstring()
{
	index_t start=0;
	index_t end=0;

	end=m_tokenizer->next_token_idx(start);

	SGVector<char> result=SGVector<char>(end-start+1);
	for (index_t i=start; i<end; i++)
	{
		result[i-start]=m_text[i];
	}
	result[end-start]='\0';

	return result;
}

bool Parser::read_bool()
{
	SGVector<char> token=read_cstring();

	if (token.vlen>0)
		return (bool) strtod(token.vector, NULL);
	else
		return (bool) 0L;
}

#define READ_INT_METHOD(fname, convf, sg_type) \
sg_type Parser::fname() \
{ \
	SGVector<char> token=read_cstring(); \
	\
	if (token.vlen>0) \
		return (sg_type) convf(token.vector, NULL, 10); \
	else \
		return (sg_type) 0L; \
}

READ_INT_METHOD(read_long, strtoll, int64_t)
READ_INT_METHOD(read_ulong, strtoull, uint64_t)
#undef READ_INT_METHOD

#define READ_REAL_METHOD(fname, convf, sg_type) \
sg_type Parser::fname() \
{ \
	SGVector<char> token=read_cstring(); \
	\
	if (token.vlen>0) \
		return (sg_type) convf(token.vector, NULL); \
	else \
		return (sg_type) 0L; \
}

READ_REAL_METHOD(read_char, strtod, char)
READ_REAL_METHOD(read_byte, strtod, uint8_t)
READ_REAL_METHOD(read_short, strtod, int16_t)
READ_REAL_METHOD(read_word, strtod, uint16_t)
READ_REAL_METHOD(read_int, strtod, int32_t)
READ_REAL_METHOD(read_uint, strtod, uint32_t)

READ_REAL_METHOD(read_short_real, strtod, float32_t)
READ_REAL_METHOD(read_real, strtod, float64_t)
#ifdef HAVE_STRTOLD
READ_REAL_METHOD(read_long_real, strtold, floatmax_t)
#else
READ_REAL_METHOD(read_long_real, strtod, floatmax_t)
#endif
#undef READ_REAL_METHOD

void Parser::set_text(SGVector<char> text)
{
	m_text=text;

	if (m_tokenizer!=NULL)
		m_tokenizer->set_text(m_text);
}

void Parser::set_tokenizer(std::shared_ptr<Tokenizer> tokenizer)
{
	
	
	m_tokenizer=tokenizer;

	if (m_tokenizer!=NULL)
		m_tokenizer->set_text(m_text);
}

void Parser::init()
{
	m_text=SGVector<char>();
	m_tokenizer=NULL;
}
