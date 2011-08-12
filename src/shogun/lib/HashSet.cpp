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

#include <shogun/lib/HashSet.h>

using namespace shogun;

CHashSet::CHashSet()
{
	array_size = 0;
	hash_array = NULL;
}

CHashSet::CHashSet(int32_t size)
{
	array_size = size;
	hash_array = SG_MALLOC(HashSetNode*, array_size);
	for(int32_t i = 0; i < array_size; i++)
	{
		hash_array[i] = NULL;
	}
}

CHashSet::~CHashSet()
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

bool CHashSet::insert_key(int32_t key, float64_t data)
{
	int32_t index = hash(key);
	if(chain_search(index, key) != NULL)
	{
		// this key is already in set
		return false;
	}

	// init new node
	HashSetNode* new_node = new HashSetNode;
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

bool CHashSet::search_key(int32_t key, float64_t &ret_data)
{
	int index = hash(key);

	HashSetNode* result = chain_search(index, key);
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

HashSetNode* CHashSet::chain_search(int32_t index, int32_t key)
{
	if(hash_array[index] == NULL)
	{
		return NULL;
	}
	else
	{
		HashSetNode* current = hash_array[index];


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

void CHashSet::delete_key(int32_t key)
{
	int index = hash(key);
	HashSetNode* result = chain_search(index, key);

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

void CHashSet::debug()
{
	for(int32_t i = 0; i < array_size; i++)
	{
		HashSetNode* current = hash_array[i];

		if(current == NULL)
		{
			SG_SPRINT("NULL\n");
			continue;
		}

		do
		{
			SG_SPRINT("%d ", current->key);
			current = current->right;
		}
		while(current != NULL);
		printf("\n");
	}
}

int32_t CHashSet::hash(int32_t key)
{
	return CHash::MurmurHash2((uint8_t*)(&key), 4, 1) % array_size;
}

