/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _DELIMITERTOKENIZER__H__
#define	_DELIMITERTOKENIZER__H__

#include <shogun/lib/config.h>

#include <shogun/lib/Tokenizer.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{

/** @brief The class CDelimiterTokenizer is used to tokenize
 *  a SGVector<char> into tokens using custom chars as delimiters.
 *  One can set the delimiters to use by setting to 1 the appropriate
 *  index of the public field delimiters. Eg. to set as delimiter the
 *  character ':', one should do: tokenizer->delimiters[':'] = 1;
 */
class CDelimiterTokenizer: public CTokenizer
{
public:
	/** default constructor
	 *
	 * @param skip_delimiters whether to skip consecutive delimiters or not
	 */
	CDelimiterTokenizer(bool skip_delimiters = false);

	/** copy constructor
	 *
	 * @param orig the original DelimiterTokenizer
	 */
	CDelimiterTokenizer(const CDelimiterTokenizer& orig);

	/** destructor */
	virtual ~CDelimiterTokenizer() {}

	/** Set the char array that requires tokenization
	 *
	 * @param txt the text to tokenize
	 */
	virtual void set_text(SGVector<char> txt);

	/** Returns true or false based on whether
	 * there exists another token in the text
	 *
	 * @return if another token exists
	 */
	virtual bool has_next();

	/** Method that returns the indices, start and end, of
	 *  the next token in line.
	 *  If next_token starts with a delimiter and skip_consecutive_delimiters is false,
	 *  it returns the same indices for start and end.
	 *
	 * @param start token's starting index
	 * @return token's ending index (exclusive)
	 */
	virtual index_t next_token_idx(index_t& start);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 * the CLASS NAME without the prefixed 'C'.
	 *
	 * @return name of the SGSerializable
	 */
	virtual const char* get_name() const;

	/** Makes the tokenizer to use ' ' or '\\t'
	 *  as the delimiters for the tokenization process;
	 */
	void init_for_whitespace();

	CDelimiterTokenizer* get_copy();

	/** Resets the delimiters */
	void clear_delimiters();

	/** Get skip_consecutive_delimiters
	 *
	 * @return if skip consecutive delimiters is set
	 */
	bool get_skip_delimiters() const;

	/** set value for skip_consecutive_delimiters
	 *
	 * @param skip_delimiters whether to skip or not consecutive delimiters
	 */
	void set_skip_delimiters(bool skip_delimiters);

private:
	void init();

public:
	/** delimiters */
	SGVector<bool> delimiters;

protected:
	/** index of last token */
	index_t last_idx;

	/** whether to skip consecutive delimiters or not */
	bool skip_consecutive_delimiters;
};
}
#endif	/* _WHITESPACETOKENIZER__H__ */

