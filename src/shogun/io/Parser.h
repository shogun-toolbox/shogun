/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#ifndef __PARSER_H__
#define __PARSER_H__

#include <shogun/lib/SGVector.h>
#include <shogun/lib/Tokenizer.h>

namespace shogun
{
/** @brief Class for reading from a string */
class CParser : public CSGObject
{
public:
	/** default constructor */
	CParser();

	/** constructor
	 *
	 * @param string the text to parse
	 * @param tokenizer tokenizer
	 */
	CParser(SGVector<char> string, CTokenizer* tokenizer);

	/** destructor */
	virtual ~CParser();

	/** check for next line in the stream
	 *
	 * @return true if there is next line, false - otherwise
	 */
	virtual bool has_next();

	/** skip next token */
	virtual void skip_token();

	/** read string	*/
	virtual SGVector<char> read_string();

	/** read one of the several base data types. */
	//@{
	virtual bool read_bool();
	virtual char read_char(int32_t base=10);
	virtual uint8_t read_byte(int32_t base=10);
	virtual int16_t read_short(int32_t base=10);
	virtual uint16_t read_word(int32_t base=10);
	virtual int32_t read_int(int32_t base=10);
	virtual uint32_t read_uint(int32_t base=10);
	virtual int64_t read_long(int32_t base=10);
	virtual uint64_t read_ulong(int32_t base=10);
	virtual float32_t read_short_real();
	virtual float64_t read_real();
	virtual floatmax_t read_long_real();
	//@}

	/** set tokenizer
	 *
	 * @param tokenizer tokenizer	
	 */
	void set_tokenizer(CTokenizer* tokenizer);

	/** set the char array that requires tokenization
	 *
	 * @param txt the text to tokenize
	 */
	void set_text(SGVector<char> text);

	/** @return object name */
	virtual const char* get_name() const { return "Parser"; }

private:
	/** class initialization */
	void init();

private:
	/** text to tokenizer */
	SGVector<char> m_text;

	/** tokenizer */
	CTokenizer* m_tokenizer;
};

}

#endif /** __STRING_READER_H__ */
