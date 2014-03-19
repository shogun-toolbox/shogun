/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _NGRAMTOKENIZER__H__
#define	_NGRAMTOKENIZER__H__

#include <shogun/lib/Tokenizer.h>

namespace shogun
{
template <class T> class SGVector;

/** @brief The class CNGramTokenizer is used to tokenize
 *  a SGVector<char> into n-grams
 */
class CNGramTokenizer: public CTokenizer
{

public:
    /** Constructor
	 *
	 * @param ns N-grams' size
	 */
    CNGramTokenizer(int32_t ns=3);

    /** copy constructor
	 *
	 * @param orig the original NGramTokenizer
	 */
    CNGramTokenizer(const CNGramTokenizer& orig);

    /** destructor */
    virtual ~CNGramTokenizer() {}

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
	 *
	 * @param start token's starting index
	 * @return token's ending index (exclusive)
	 */
	virtual index_t next_token_idx(index_t& start);

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
    virtual const char* get_name() const;

	virtual CNGramTokenizer* get_copy();

private:
	void init();

protected:

	/** n-grams' size */
	int32_t n;

	/** last index returned */
	index_t last_idx;
};
}
#endif	/* _NGRAMTOKENIZER__H__ */

