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

#ifndef _SET_H_
#define _SET_H_

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Hash.h>
#include <shogun/base/DynArray.h>

#include <cstdio>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** hashset node */
template<class T> struct CSetNode
{	
	/** index in hashtable */
	int32_t index;

	/** is free? */
	bool free;

	/** data of node */
	T element;

	/** pointer to left sibling */
	CSetNode<T> *left;

	/** pointer to right sibling */
	CSetNode<T> *right;
};
#endif

/** @brief the class CSet, a set based on the hash-table.
 * w: http://en.wikipedia.org/wiki/Hash_table
 */
template<class T> class CSet: public CSGObject
{
public:
	/** Custom constructor */
	CSet(int32_t size=41, int32_t reserved=128, bool tracable=true)
	{	
		hash_size=size;
		free_index=0;
		num_elements=0;
		use_sg_mallocs=tracable;

		if (use_sg_mallocs)
			hash_array=SG_CALLOC(CSetNode<T>*, size);
		else
			hash_array=(CSetNode<T>**) calloc(size, sizeof(CSetNode<T>*));

		for (int32_t i=0; i<size; i++)
		{
			hash_array[i]=NULL;
		}

		array=new DynArray<CSetNode<T>* >(reserved, tracable);
	}

	/** Default destructor */
	virtual ~CSet()
	{
		if (array!=NULL)
		{
			for(int32_t i=0; i<array->get_num_elements(); i++)
			{
				if (array->get_element(i)!=NULL)
				{
					if (use_sg_mallocs)
						SG_FREE(array->get_element(i));
					else
						free(array->get_element(i));
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

	/** @return object name */
	virtual const char* get_name() const { return "Set"; }

	/** Add an element to the set
	 *
	 * @param element element to be added
	 */
	void add(const T& element)
	{
		int32_t index=hash(element);
		if (chain_search(index, element)==NULL)
		{
			insert_key(index, element);
			num_elements++;
		}
	}

	/** Remove an element from the set
	 *
	 * @param element element to be looked for
	 */
	bool contains(const T& element)
	{
		int32_t index=hash(element);
		if (chain_search(index, element)!=NULL)
			return true; 

		return false;
	}

	/** Remove an element from the set
	 *
	 * @param element element to be removed
	 */
	void remove(const T& element)
	{
		int32_t index=hash(element);
		CSetNode<T>* result=chain_search(index, element);

		if (result!=NULL)		
		{
			delete_key(index, result);
			num_elements--;
		}
	}

	/** Index of element in the set
	 *
	 * @param element element to be looked for
	 * @return index of the element or -1 if not found
	 */
	int32_t index_of(const T& element)
	{
		int32_t index=hash(element);
		CSetNode<T>* result=chain_search(index, element);

		if (result!=NULL)
			 return result->index;
		
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

	/** Get size of auxilary array
	 *
	 * @return array size
	 */
	int32_t get_array_size() const
	{
		return array->get_num_elements();
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
		if (array->get_element(index)!=NULL)
			return &(array->get_element(index)->element);
		return NULL;
	}

	/** get node at index as reference
	 *
	 * (does NOT do bounds checking)
	 *
	 * @param index index
	 * @return node at index
	 */
	CSetNode<T>* get_node_ptr(int32_t index)
	{
		return array->get_element(index);
	}
		
	/** @return underlying array of nodes in memory */
	CSetNode<T>** get_array()
	{
		return array->get_array();
	}

	/** get element
	 *
	 * (does NOT do bounds checking)
	 *
	 * @param index index
	 * @return element  
	 */
	T get_element(int32_t index)
	{
		if (array->get_element(index)!=NULL)
			return array->get_element(index)->element;
		return T();
	}

private:
	/** Returns hash of key
	 * MurmurHash used
	 */
	int32_t hash(const T& element)
	{
		return CHash::MurmurHash3((uint8_t*)(&element), sizeof(element), 0xDEADBEEF) % hash_size;
	}

	/** is free? */
	bool is_free(CSetNode<T>* node)
	{
		if (node->free==true)
			return true;

		return false;
	}

	/** Searchs key in list(chain) */
	CSetNode<T>* chain_search(int32_t index, const T& element)
	{
		if (hash_array[index]==NULL)
		{
			return NULL;
		}
		else
		{
			CSetNode<T>* current=hash_array[index];

			do // iterating all items in the list
			{
				if (current->element==element)
					return current; // it's a search key

				current=current->right;

			} while (current!=NULL);

			return NULL;
		}
	}
	
	/** Inserts nodes with certain key and data in set */
	void insert_key(int32_t index, const T& element)
	{
		int32_t new_index;
		CSetNode<T>* new_node;

		if ((free_index>=array->get_num_elements()) || (array->get_element(free_index)==NULL))
		{
			// init new node
			if (use_sg_mallocs)
				new_node=SG_CALLOC(CSetNode<T>, 1);
			else
				new_node=(CSetNode<T>*) calloc(1, sizeof(CSetNode<T>));

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
		new_node->element=element;
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
	void delete_key(int32_t index, CSetNode<T>* node)
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
	CSetNode<T>** hash_array;

	/** array for index permission */
	DynArray<CSetNode<T>*>* array;
};

}

#endif /* _MAP_H_ */
