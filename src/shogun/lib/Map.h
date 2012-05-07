/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Evgeniy Andreev (gsomix)
 *
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

#ifndef _MAP_H_
#define _MAP_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Hash.h>
#include <shogun/base/DynArray.h>

#include <cstdio>

namespace shogun
{

#define IGNORE_IN_CLASSLIST

#define MapNode CMapNode<K, T>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** hashset node */
IGNORE_IN_CLASSLIST template<class K, class T> struct CMapNode
{	
	/** index in hashtable */
	int32_t index;

	/** is free? */
	bool free;

	/** key of node */
	K key;

	/** data of node */
	T data;

	/** pointer to left sibling */
	CMapNode *left;

	/** pointer to right sibling */
	CMapNode *right;
};
#endif

/** @brief the class CSet, a set based on the hash-table.
 * w: http://en.wikipedia.org/wiki/Hash_table
 */
IGNORE_IN_CLASSLIST template<class K, class T> class CMap: public CSGObject
{
public:
	/** Default constructor */
	CMap()
	{	
		hash_size=0;
		free_index=0;
		num_elements=0;
		hash_array=NULL;
		array=NULL;
		use_sg_mallocs=false;
	}

	/** Custom constructor */
	CMap(int32_t size, int32_t reserved=1024, bool tracable=true)
	{	
		hash_size=size;
		free_index=0;
		num_elements=0;
		use_sg_mallocs=tracable;

		if(use_sg_mallocs)
		{
			hash_array=SG_CALLOC(MapNode*, size);
		}
		else
		{
			hash_array=(CMapNode<K, T>**) calloc(size, sizeof(CMapNode<K, T>*));
		}

		for (int32_t i=0; i<size; i++)
		{
			hash_array[i]=NULL;
		}

		array=new DynArray<CMapNode<K, T>*>(reserved, tracable);
	}

	/** Default destructor */
	virtual ~CMap()
	{
		if (array!=NULL)
		{
			for(int32_t i=0; i<array->get_num_elements(); i++)
			{
				if(array->get_element(i)!=NULL)
				{
					if(use_sg_mallocs)
					{
						SG_FREE(array->get_element(i));
					}
					else
					{
						free(array->get_element(i));
					}
				}
			}
			delete array;
		}

		if (hash_array!=NULL)
		{
			if(use_sg_mallocs)
			{
				SG_FREE(hash_array);
			}
			else
			{
				free(hash_array);
			}
		}
	}

	/** @return object name */
	virtual const char* get_name() const { return "Map"; }

	/** Add an element to the set
	 *
	 * @param e elemet to be added
	 */
	void add(const K& key, const T& data)
	{
		int32_t index=hash(key);
		if (chain_search(index, key)==NULL)
		{
			insert_key(index, key, data);
			num_elements++;
		}
	}

	/** Remove an element from the set
	 *
	 * @param e element to be looked for
	 */
	bool contains(const K& key)
	{
		int32_t index=hash(key);
		if (chain_search(index, key)!=NULL)
		{
			return true; 
		}

		return false;
	}

	/** Remove an element from the set
	 *
	 * @param e element to be removed
	 */
	void remove(const K& key)
	{
		int32_t index=hash(key);
		CMapNode<K, T>* result=chain_search(index, key);

		if (result!=NULL)		
		{
			delete_key(index, result);
			num_elements--;
		}
	}

	/** Index of element in the set
	 *
	 * @param e element to be removed
	 * @return index of the element or -1 if not found
	 */
	int32_t index_of(const K& key)
	{
		int32_t index=hash(key);
		CMapNode<K ,T>* result=chain_search(index, key);

		if (result!=NULL)		
		{
			 return result->index;
		}
		
		return -1;
	}

	/** Get number of elements
	 *
	 * @return number of elements
	 */
	int32_t get_num_elements() const
	{
		return num_elements;
	}

	/** Get set element at index
	 *
	 * (does NOT do bounds checking)
	 *
	 * @param index index
	 * @return array element at index
	 */
	T get_element(int32_t index) const
	{
		return array->get_element(index)->data;
	}

	/** get set element at index as reference
	 *
	 * (does NOT do bounds checking)
	 *
	 * @param index index
	 * @return array element at index
	 */
	T* get_element_ptr(int32_t index)
	{
		if (is_free(array->get_element(index)))
			return NULL;
		return &(array->get_element(index)->data);
	}

	/** operator overload for set read only access
	 * use add() for write access
	 *
	 * DOES NOT DO ANY BOUNDS CHECKING
	 *
	 * @param index index
	 * @return element at index
	 */
	T operator[](int32_t index) const
	{
		return array->get_element(index)->data;
	}
		
	/** @return underlying array of nodes in memory */
	T* get_array()
	{
		return array->get_array();
	}

private:
	/** Returns hash of key
	 * MurmurHash used
	 */
	int32_t hash(const K& key)
	{
		return CHash::MurmurHash2((uint8_t*)(&key), sizeof(key), 0xDEADBEEF) % hash_size;
	}

	/** is free? */
	bool is_free(CMapNode<K, T>* node)
	{
		if (node->free==true)
		{
			return true;
		}

		return false;
	}

	/** Searchs key in list(chain) */
	CMapNode<K, T>* chain_search(int32_t index, const K& key)
	{
		if (hash_array[index]==NULL)
		{
			return NULL;
		}
		else
		{
			CMapNode<K, T>* current=hash_array[index];

			do // iterating all items in the list
			{
				if (current->key==key)
				{
					return current; // it's a search key
				}

				current=current->right;

			} while (current!=NULL);

			return NULL;
		}
	}
	
	/** Inserts nodes with certain key and data in set */
	void insert_key(int32_t index, const K& key, const T& data)
	{
		int32_t new_index;
		CMapNode<K, T>* new_node;

		if ((free_index>=array->get_num_elements()) || (array->get_element(free_index)==NULL))
		{
			// init new node
			if(use_sg_mallocs)
			{
				new_node=SG_CALLOC(MapNode, 1);
			}
			else
			{
				new_node=(CMapNode<K, T>*) calloc(1, sizeof(CMapNode<K, T>));
			}

			array->append_element(new_node);

			new_index=free_index;
			free_index++;
		}
		else
		{
			new_node=array->get_element(free_index);
			ASSERT(is_free(new_node));

			new_index=free_index;
			free_index=new_node->index;
		}

		new_node->index=new_index;
		new_node->free=false;
		new_node->key=key;
		new_node->data=data;
		new_node->left=NULL;
		new_node->right=NULL;

		// add new node in start of list
		if (hash_array[index]==NULL)
		{
			hash_array[index]=new_node;
		}
		else
		{
			hash_array[index]->left=new_node;
			new_node->right=hash_array[index];

			hash_array[index]=new_node;
		}
	}

	/** Deletes key from set */
	void delete_key(int32_t index, CMapNode<K, T>* node)
	{		
		int32_t temp=0;

		if (node==NULL)
		{
			return;
		}

		if (node->right!=NULL)
		{
			node->right->left = node->left;
		}

		if (node->left!=NULL)
		{
			node->left->right = node->right;		
		}
		else
		{
			hash_array[index] = node->right;
		}

		temp=node->index;

		node->index=free_index;
		node->free=true;
		node->left=NULL;
		node->right=NULL;

		free_index=temp;		
	}


protected:
	/** whether SG_MALLOC or just malloc etc shall be used */
	bool use_sg_mallocs;

	/** hashtable size */
	int32_t hash_size;

	/** next free index for new element */
	int32_t free_index;

	/** number of elements */
	int32_t num_elements;

	/** array of lists (chains) */
	CMapNode<K, T>** hash_array;

	/** array for index permission */
	DynArray<CMapNode<K, T>*>* array;
};

}

#endif /* _MAP_H_ */
