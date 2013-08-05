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

#include <shogun/io/SGIO.h>
#include <shogun/lib/Lock.h>


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

/** @brief the class CMap, a map based on the hash-table.
 * w: http://en.wikipedia.org/wiki/Hash_table
 */
IGNORE_IN_CLASSLIST template<class K, class T> class CMap: public CSGObject
{
public:
	/** Custom constructor */
	CMap(int32_t size=41, int32_t reserved=128, bool tracable=true)
	{
		hash_size=size;
		free_index=0;
		num_elements=0;
		use_sg_mallocs=tracable;

		if (use_sg_mallocs)
			hash_array=SG_CALLOC(MapNode*, size);
		else
			hash_array=(CMapNode<K, T>**) calloc(size, sizeof(CMapNode<K, T>*));

		for (int32_t i=0; i<size; i++)
		{
			hash_array[i]=NULL;
		}

		array=new DynArray<CMapNode<K, T>*>(reserved, tracable);
	}

	/** Default destructor */
	virtual ~CMap()
	{
		destroy_map();
	}

	/** @return object name */
	virtual const char* get_name() const { return "Map"; }

	/** Add an element to the map
	 *
	 * @param key key to be added
	 * @param data data to be added
	 * @return index of added element
	 */
	int32_t add(const K& key, const T& data)
	{
		int32_t index=hash(key);
		if (chain_search(index, key)==NULL)
		{
			lock.lock();
			int32_t added_index=insert_key(index, key, data);
			num_elements++;
			lock.unlock();

			return added_index;
		}

		return -1;
	}

	/** Check an element in the map
	 *
	 * @param key key to be looked for
	 * @return true if element contains in the map
	 */
	bool contains(const K& key)
	{
		int32_t index=hash(key);
		if (chain_search(index, key)!=NULL)
			return true;

		return false;
	}

	/** Remove an element from the set
	 *
	 * @param key key to be removed
	 */
	void remove(const K& key)
	{
		int32_t index=hash(key);
		CMapNode<K, T>* result=chain_search(index, key);

		if (result!=NULL)
		{
			lock.lock();
			delete_key(index, result);
			num_elements--;
			lock.unlock();
		}
	}

	/** Index of element in the set
	 *
	 * @param key key to be looked for
	 * @return index of the element or -1 if not found
	 */
	int32_t index_of(const K& key)
	{
		int32_t index=hash(key);
		CMapNode<K ,T>* result=chain_search(index, key);

		if (result!=NULL)
			return result->index;

		return -1;
	}

	/** Get element by key
	 *
	 * @param key key to be looked for
	 * @return exist element or new element
	 * (if key doesn't consist in map)
	 */
	T get_element(const K& key)
	{
		int32_t index=hash(key);
		CMapNode<K, T>* result=chain_search(index, key);

		if (result!=NULL)		
			return result->data;
		else
		{
			int32_t added_index=add(key, T());
			result=get_node_ptr(added_index);

			return result->data;
		}
	}

	/** Set element by key
	 *
	 * @param key key of element
	 * @param data new data for element
	 */
	void set_element(const K& key, const T& data)
	{
		int32_t index=hash(key);
		CMapNode<K, T>* result=chain_search(index, key);

		lock.lock();

		if (result!=NULL)
			result->data=data;
		else
			add(key, data);

		lock.unlock();
	}

	/** Get number of elements
	 *
	 * @return number of elements
	 */
	int32_t get_num_elements() const
	{
		return num_elements;
	}

	/** Get size of auxilary array
	 *
	 * @return array size
	 */
	int32_t get_array_size() const
	{
		return array->get_num_elements();
	}

	/** get element at index as reference
	 *
	 * (does NOT do bounds checking)
	 *
	 * @param index index
	 * @return array element at index
	 */
	T* get_element_ptr(int32_t index)
	{
		CMapNode<K, T>* result=array->get_element(index);
		if (result!=NULL && !is_free(result))
			return &(array->get_element(index)->data);
		return NULL;
	}

	/** get node at index as reference
	 *
	 * (does NOT do bounds checking)
	 *
	 * @param index index
	 * @return node at index
	 */
	CMapNode<K, T>* get_node_ptr(int32_t index)
		{
		return array->get_element(index);
		}

	/** @return underlying array of nodes in memory */
	CMapNode<K, T>** get_array()
		{
		return array->get_array();
		}

	/** assignment operator that copies map */
	CMap& operator =(const CMap& orig)
	{

		destroy_map();
		use_sg_mallocs = orig.use_sg_mallocs;

		hash_size = orig.hash_size;

		if (use_sg_mallocs)
			hash_array = SG_CALLOC(MapNode*, hash_size);
		else
			hash_array = (CMapNode<K, T>**) calloc(hash_size,
					sizeof(CMapNode<K, T>*));

		for (int32_t i = 0; i<hash_size; i++)
		{
			hash_array[i] = NULL;
		}

		array = new DynArray<CMapNode<K, T>*>(128, use_sg_mallocs);

		for (int i = 0; i < orig.num_elements; i++)
		{
			CMapNode<K, T>* node = orig.array->get_element(i);
			add(node->key, node->data);
		}

		return *this;
	}

	/** Get or set element by key
	 *
	 * @param key key to be looked for
	 * @return reference exist element or new element
	 */
	T& operator [](const K& key)
	{
		int32_t index=hash(key);
		CMapNode<K, T>* result=chain_search(index, key);

		if (result!=NULL)		
			return result->data;
		else
		{
			int32_t added_index=add(key, T());
			result=get_node_ptr(added_index);

			return result->data;
		}
	}

private:
	/** Returns hash of key
	 * MurmurHash used
	 */
	int32_t hash(const K& key)
	{
		return CHash::MurmurHash3((uint8_t*)(&key), sizeof(key), 0xDEADBEEF) % hash_size;
	}

	/** is free? */
	bool is_free(CMapNode<K, T>* node)
	{
		if (node->free==true)
			return true;

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
					return current; // it's a search key

				current=current->right;

			} while (current!=NULL);

			return NULL;
		}
		}

	/** Inserts nodes with certain key and data in set */
	int32_t insert_key(int32_t index, const K& key, const T& data)
	{
		int32_t new_index;
		CMapNode<K, T>* new_node;

		if ((free_index>=array->get_num_elements()) || (array->get_element(free_index)==NULL))
		{
			// init new node
			if (use_sg_mallocs)
				new_node=SG_CALLOC(MapNode, 1);
			else
				new_node=(CMapNode<K, T>*) calloc(1, sizeof(CMapNode<K, T>));

			new (&new_node->key) K();
			new (&new_node->data) T();
	
			array->append_element(new_node);

			new_index=free_index;
			free_index++;
		}
		else
		{
			new_node=array->get_element(free_index);
			ASSERT(is_free(new_node))

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

		return new_index;
	}

	/** Deletes key from set */
	void delete_key(int32_t index, CMapNode<K, T>* node)
	{
		int32_t temp=0;

		if (node==NULL)
			return;

		if (node->right!=NULL)
			node->right->left = node->left;

		if (node->left!=NULL)
			node->left->right = node->right;
		else
			hash_array[index] = node->right;

		temp=node->index;

		node->index=free_index;
		node->free=true;
		node->left=NULL;
		node->right=NULL;

		free_index=temp;
	}

	/*cleans up map*/
	void destroy_map()
	{
		if (array!=NULL)
		{
			for(int32_t i=0; i<array->get_num_elements(); i++)
			{
				CMapNode<K, T>* element = array->get_element(i);
				if (element!=NULL)
				{
					element->key.~K();
					element->data.~T();

					if (use_sg_mallocs)
						SG_FREE(element);
					else
						free(element);
				}
			}
			delete array;
		}

		if (hash_array!=NULL)
		{
			if (use_sg_mallocs)
				SG_FREE(hash_array);
			else
				free(hash_array);
		}

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

	/** concurrency lock */
	CLock lock;
};

}

#endif /* _MAP_H_ */
