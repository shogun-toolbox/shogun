/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#ifndef _HASHEDDOCCONVERTER__H__
#define _HASHEDDOCCONVERTER__H__

#include <shogun/converter/Converter.h>
#include <shogun/features/Features.h>
#include <shogun/lib/Tokenizer.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
class CFeatures;
class CTokenizer;
class CConverter;
template<class T> class CSparseFeatures;

/** @brief This class can be used to convert a document collection contained in a CStringFeatures<char>
 * object where each document is stored as a single vector into a hashed Bag-of-Words representation.
 * Like in the standard Bag-of-Words representation, this class considers each document as a collection of tokens,
 * which are then hashed into a new feature space of a specified dimension.
 * This class is very flexible and allows the user to specify the tokenizer used to tokenize each document,
 * specify whether the results should be normalized with regards to the sqrt of the document size, as well
 * as to specify whether he wants to combine different tokens.
 * The latter implements a k-skip n-grams approach, meaning that you can combine up to n tokens, while skipping up to k.
 * Eg. for the tokens ["a", "b", "c", "d"], with n_grams = 2 and skips = 2, one would get the following combinations :
 * ["a", "ab", "ac" (skipped 1), "ad" (skipped 2), "b", "bc", "bd" (skipped 1), "c", "cd", "d"].
 */
class CHashedDocConverter : public CConverter
{
public:
	/** Default constructor */
	CHashedDocConverter();

	/** Constructor
	 * Creates tokens on whitespace
	 *
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param normalize whether to normalize vectors or not
	 * @param n_grams the max number of tokens to consider when combining tokens
	 * @param skips the max number of tokens to skip when combining tokens
	 */
	CHashedDocConverter(int32_t hash_bits, bool normalize = false, int32_t n_grams = 1, int32_t skips = 0);

	/** Constructor
	 *
	 * @param tzer the tokenizer to use
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param normalize whether to normalize vectors or not
	 * @param n_grams the max number of tokens to consider when combining tokens
	 * @param skips the max number of tokens to skip when combining tokens
	 */
	CHashedDocConverter(CTokenizer* tzer, int32_t hash_bits, bool normalize = false, int32_t n_grams = 1,
		int32_t skips = 0);

	/** Destructor */
	virtual ~CHashedDocConverter();

	/** Hashes each string contained in features 
	 *
	 * @param features the strings to be hashed. Must be an instance of CStringFeatures.
	 * @return a CSparseFeatures object containing the hashes of the strings.
	 */
	virtual CFeatures* apply(CFeatures* features);

	/** Hashes the tokens contained in document
	 *
	 * @param document the char vector to tokenize and hash 
	 * @return a SGSparseVector with the hashed representation of the document
	 */
	SGSparseVector<float64_t> apply(SGVector<char> document);

	/** Generate all the k-skip ngram combinations for the tokens in the CDynArray hashes
	 * and then hash limit them to a specific dimension
	 *
	 * @param hashes the hashes of the tokens to combine as k-skip ngrams
	 * @param ngram_hashes the dynamic array in which to store the created indices
	 * @param num_bits the dimension in which to limit the hashed indices (means a dimension of size 2^num_bits) 
	 * @param ngrams the max number of tokens to combine
	 * @param tokens_to_skip the max number of tokens to skip when combining (starting always from the second one)
	 * @return an array containing the hashed indices of the combined tokens
	 */
	static void generate_ngram_hashes(CDynamicArray<uint32_t>* hashes, CDynamicArray<index_t>* ngram_hashes,
			int32_t num_bits, int32_t ngrams, int32_t tokens_to_skip);

	/** @return object name */
	virtual const char* get_name() const;

	/** specify whether hashed vector should be normalized or not
	 *
	 * @param normalize  whether to normalize
	 */
	void set_normalization(bool normalize);

	/** Method used to specify the parameters for the quadratic
	 * approach of k-skip n-grams. See class description for more 
	 * details and an example.
	 *
	 * @param k the max number of allowed skips
	 * @param n the max number of tokens to combine
	 */
	void set_k_skip_n_grams(int32_t k, int32_t n);
protected:
	
	/** init */
	void init(CTokenizer* tzer, int32_t d, bool normalize, int32_t n_grams, int32_t skips);

	/** This method takes a dynamic array as an argument, sorts it and returns the number
	 * of the distinct elements(indices here) in the array.
	 *
	 * @param hashed_indices the array to sort and count elements
	 * @return the number of distinct elements
	 */
	int32_t count_distinct_indices(CDynamicArray<uint32_t>& hashed_indices);

	/** This method takes the dynamic array containing all the hashed indices of a document and returns a compact
	 * sparse representation with each index found and with the count of such index
	 * 
	 * @param hashed_indices the array containing the hashed indices
	 * @return the compact hashed document representation
	 */
	SGSparseVector<float64_t> create_hashed_representation(CDynamicArray<uint32_t>& hashed_indices);

protected:

	/** the number of bits of the hash */
	int32_t num_bits;

	/** the tokenizer */
	CTokenizer* tokenizer;

	/** whether to normalize or not */
	bool should_normalize;

	/** the number of consecutives tokens for quadratic */
	int32_t ngrams;

	/** the number of tokens to skip */
	int32_t tokens_to_skip;
};
}

#endif
