/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Evgeniy Andreev (gsomix)
 *
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef HASHSET_H_
#define HASHSET_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Hash.h>

namespace shogun
{

struct HashSetNode
{
	/** key of node */
	int32_t key;

	/** data of node */
	float64_t data;

	/** pointer to left sibling */
	HashSetNode *left;

	/** pointer to right sibling */
	HashSetNode *right;
};

/** @brief the class HashSet, a set based on the hash-table.
 * w: http://en.wikipedia.org/wiki/Hash_table
 */
class CHashSet: public CSGObject
{
public:
	CHashSet();

	/** Constructor for heap with specified size of hash array */
	CHashSet(int32_t size);

	virtual inline const char* get_name() const
	{
		return "HashSet";
	}

	virtual ~CHashSet();

	/** Inserts nodes with certain key and data in set */
	bool insert_key(int32_t key, float64_t data);

	/** Searchs data by key in set */
	bool search_key(int32_t key, float64_t &ret_data);

	/** Deletes key from set */
	void delete_key(int32_t key);

	/** Debug "pretty" print */
	void debug();
private:
	/** Returns hash of key
	 * MurmurHash used
	 */
	int32_t hash(int32_t key);

	/** Searchs key in list(chain) */
	HashSetNode* chain_search(int32_t index, int32_t key);

protected:
	/** array of lists(chains) */
	HashSetNode **hash_array;

	/** size of array */
	int32_t array_size;
};

}

#endif /* HASHSET_H_ */
