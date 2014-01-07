/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <stdlib.h>
#include <io/Parser.h>

using namespace shogun;

CParser::CParser()
{
	init();
}

CParser::CParser(SGVector<char> text, CTokenizer* tokenizer)
{
	init();

	m_text=text;

	SG_REF(tokenizer);
	m_tokenizer=tokenizer;

	if (m_tokenizer!=NULL)
		m_tokenizer->set_text(m_text);
}

CParser::~CParser()
{
	SG_UNREF(m_tokenizer);
}

bool CParser::has_next()
{
	if (m_tokenizer!=NULL)
		return m_tokenizer->has_next();

	return false;
}

void CParser::skip_token()
{
	index_t start=0;
	m_tokenizer->next_token_idx(start);
}

SGVector<char> CParser::read_string()
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

SGVector<char> CParser::read_cstring()
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

bool CParser::read_bool()
{
	SGVector<char> token=read_cstring();

	if (token.vlen>0)
		return (bool) strtod(token.vector, NULL);
	else
		return (bool) 0L;
}

#define READ_INT_METHOD(fname, convf, sg_type) \
sg_type CParser::fname() \
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
sg_type CParser::fname() \
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

void CParser::set_text(SGVector<char> text)
{
	m_text=text;

	if (m_tokenizer!=NULL)
		m_tokenizer->set_text(m_text);
}

void CParser::set_tokenizer(CTokenizer* tokenizer)
{
	SG_REF(tokenizer);
	SG_UNREF(m_tokenizer);
	m_tokenizer=tokenizer;

	if (m_tokenizer!=NULL)
		m_tokenizer->set_text(m_text);
}

void CParser::init()
{
	m_text=SGVector<char>();
	m_tokenizer=NULL;
}
