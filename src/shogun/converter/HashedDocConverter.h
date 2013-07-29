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
	 */
	CHashedDocConverter(int32_t hash_bits, bool normalize = false);

	/** Constructor
	 *
	 * @param tzer the tokenizer to use
	 * @param hash_bits the number of bits of the hash. Means a dimension of size 2^(hash_bits).
	 * @param normalize whether to normalize vectors or not
	 */
	CHashedDocConverter(CTokenizer* tzer, int32_t hash_bits, bool normalize = false);

	/** Destructor */
	virtual ~CHashedDocConverter();

	/** Hashes each string contained in features 
	 *
	 * @param features the strings to be hashed. Must be an instance of CStringFeatures.
	 * @return a CSparseFeatures object containing the hashes of the strings.
	 */
	virtual CFeatures* apply(CFeatures* features);

	SGSparseVector<float64_t> apply(SGVector<char> document);

	virtual const char* get_name() const;

protected:
	
	/** init */
	void init(CTokenizer* tzer, int32_t d, bool normalize);

protected:

	/** the number of bits of the hash */
	int32_t num_bits;

	/** the tokenizer */
	CTokenizer* tokenizer;

	/** whether to normalize or not */
	bool should_normalize;
};
}

#endif
