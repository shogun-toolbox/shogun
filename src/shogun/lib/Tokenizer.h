/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _TOKENIZER__H__
#define _TOKENIZER__H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
class CSGObject;
template<class T> class SGVector;

/** @brief The class CTokenizer acts as a base class in order
 * to implement tokenizers. Sub-classes must implement
 * the methods has_next(), next_token_idx() and get_copy().
 */
class CTokenizer: public CSGObject
{
public:
	/** Constructor */
	CTokenizer();

	/** Destructor */
	virtual ~CTokenizer() { };

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
	virtual bool has_next()=0;

	/** Method that returns the indices, start and end, of
	 *  the next token in line.
	 *
	 * @param start token's starting index
	 * @return token's ending index (inclusive)
	 */
	virtual index_t next_token_idx(index_t& start)=0;

	/** Creates a copy of the appropriate runtime
	 * instance of a CTokenizer subclass
	 * Needs to be overriden
	 */
	virtual CTokenizer* get_copy()=0;

private:
	void init();

protected:
	/** the text to parse */
	SGVector<char> text;
};
}

#endif	/* _TOKENIZER__H__ */
