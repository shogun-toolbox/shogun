/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Evgeniy Andreev (gsomix)
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

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** hashset node */
template<class T> struct HashSetNode
{
	/** key of node */
	int32_t key;

	/** data of node */
	T data;

	/** pointer to left sibling */
	HashSetNode *left;

	/** pointer to right sibling */
	HashSetNode *right;
};
#endif

/** @brief the class HashSet, a set based on the hash-table.
 * w: http://en.wikipedia.org/wiki/Hash_table
 */
template<class T> class CHashMap: public CSGObject
{
public:
	CHashMap()
	{	
		array_size = 0;
		hash_array = NULL;
	}
	/** Constructor for heap with specified size of hash array */
	CHashMap(int32_t size)
	{
		array_size = size;
		hash_array = SG_MALLOC(HashSetNode<T>*, array_size);
		for(int32_t i = 0; i < array_size; i++)
		{
			hash_array[i] = NULL;
		}
	}

	virtual inline const char* get_name() const
	{
		return "HashSet";
	}

	virtual ~CHashMap()
	{
		if(hash_array != NULL)
		{
			for(int32_t i = 0; i < array_size; i++)
			{
				delete hash_array[i];
			}
			SG_FREE(hash_array);
		}	
	}

	/** Inserts nodes with certain key and data in set */
	bool insert_key(int32_t key, T data)
	{
		int32_t index = hash(key);
		if(chain_search(index, key) != NULL)
		{
			// this key is already in set
			return false;
		}

		// init new node
		HashSetNode<T>* new_node = new HashSetNode<T>;
		new_node->key = key;
		new_node->data = data;
		new_node->left = NULL;
		new_node->right = NULL;

		// add new node in start of list
		if(hash_array[index] == NULL)
		{
			hash_array[index] = new_node;
		}
		else
		{
			hash_array[index]->left = new_node;
			new_node->right = hash_array[index];

			hash_array[index] = new_node;
		}

		return true;	
	}

	/** Searchs data by key in set */
	bool search_key(int32_t key, T &ret_data)
	{
		int index = hash(key);

		HashSetNode<T>* result = chain_search(index, key);
		if(result == NULL)
		{
			return false;
		}
		else
		{
			ret_data = result->data;
			return true;
		}
	}

	/** Deletes key from set */
	void delete_key(T key)
	{
		int index = hash(key);
		HashSetNode<T>* result = chain_search(index, key);

		if(result == NULL)
		{
			return;
		}

		if(result->right != NULL)
		{
			result->right->left = result->left;
		}

		if(result->left != NULL)
		{
			result->left->right = result->right;
		}
		else
		{
			hash_array[index] = result->right;
		}

		result->left = NULL;
		result->right = NULL;

		delete result;
	}

private:
	/** Returns hash of key
	 * MurmurHash used
	 */
	int32_t hash(int32_t key)
	{
		return CHash::MurmurHash2((uint8_t*)(&key), sizeof(int32_t), 0xDEADBEEF) % array_size;
	}

	/** Searchs key in list(chain) */
	HashSetNode<T>* chain_search(int32_t index, int32_t key)
	{
		if(hash_array[index] == NULL)
		{
			return NULL;
		}
		else
		{
			HashSetNode<T>* current = hash_array[index];


			do // iterating all items in the list
			{
				if(current->key == key)
				{
					return current; // it's a search key
				}

				current = current->right;

			} while(current != NULL);

			return NULL;
		}
	}

protected:
	/** array of lists(chains) */
	HashSetNode<T> **hash_array;

	/** size of array */
	int32_t array_size;
};

}

#endif /* HASHSET_H_ */
