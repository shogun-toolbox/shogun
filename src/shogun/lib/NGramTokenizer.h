/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Yuyu Zhang, Bjoern Esser
 */

#ifndef _NGRAMTOKENIZER__H__
#define	_NGRAMTOKENIZER__H__

#include <shogun/lib/config.h>

#include <shogun/lib/Tokenizer.h>

namespace shogun
{
template <class T> class SGVector;

/** @brief The class NGramTokenizer is used to tokenize
 *  a SGVector<char> into n-grams
 */
class NGramTokenizer: public Tokenizer
{

public:
    /** Constructor
	 *
	 * @param ns N-grams' size
	 */
    NGramTokenizer(int32_t ns=3);

    /** copy constructor
	 *
	 * @param orig the original NGramTokenizer
	 */
    NGramTokenizer(const NGramTokenizer& orig);

    /** destructor */
    ~NGramTokenizer() override {}

	/** Set the char array that requires tokenization
	 *
	 * @param txt the text to tokenize
	 */
	void set_text(SGVector<char> txt) override;

	/** Returns true or false based on whether
	 * there exists another token in the text
	 *
	 * @return if another token exists
	 */
	bool has_next() override;

	/** Method that returns the indices, start and end, of
	 *  the next token in line.
	 *
	 * @param start token's starting index
	 * @return token's ending index (exclusive)
	 */
	index_t next_token_idx(index_t& start) override;

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
    const char* get_name() const override;

	NGramTokenizer* get_copy() override;

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

