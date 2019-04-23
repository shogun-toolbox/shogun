/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Yuyu Zhang, Bjoern Esser
 */

#ifndef _TOKENIZER__H__
#define _TOKENIZER__H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGVector.h>

namespace shogun
{
class SGObject;
template<class T> class SGVector;

/** @brief The class Tokenizer acts as a base class in order
 * to implement tokenizers. Sub-classes must implement
 * the methods has_next(), next_token_idx() and get_copy().
 */
class Tokenizer: public SGObject
{
public:
	/** Constructor */
	Tokenizer();

	/** Copy constructor */
	Tokenizer(const Tokenizer& orig);

	/** Destructor */
	virtual ~Tokenizer() { };

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
	 * instance of a Tokenizer subclass
	 * Needs to be overriden
	 */
	virtual Tokenizer* get_copy()=0;

private:
	void init();

protected:
	/** the text to parse */
	SGVector<char> text;
};
}

#endif	/* _TOKENIZER__H__ */
