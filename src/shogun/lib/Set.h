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

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** hashset node */
template<class T> struct HashSetNode
{	
	/** index in set array
	 * It also using for reference to next free
	 * element in array.
	 */
	int32_t index;

	/** key and data of node */
	T element;

	/** pointer to left sibling */
	HashSetNode *left;

	/** pointer to right sibling */
	HashSetNode *right;
};
#endif

/** @brief the class CSet, a set based on the hash-table.
 * w: http://en.wikipedia.org/wiki/Hash_table
 */
template<class T> class CSet: public CSGObject
{
public:
	/** Default constructor */
	CSet()
	{	
		hash_size=0;
		free_index=0;
		num_elements=0;
		hash_array=NULL;
		array=NULL;
		use_sg_mallocs=false;
	}

	/** Custom constructor */
	CSet(int32_t size, int32_t reserved=1024, bool tracable=true)
	{	
		hash_size=size;
		free_index=0;
		num_elements=0;
		use_sg_mallocs=tracable;

		if(use_sg_mallocs)
		{
			hash_array=SG_CALLOC(HashSetNode<T>*, size);
		}
		else
		{
			hash_array=(HashSetNode<T>**) calloc(size, sizeof(HashSetNode<T>*));
		}

		for (int32_t i=0; i<size; i++)
		{
			hash_array[i]=NULL;
		}

		array=new DynArray<HashSetNode<T>*>(reserved, tracable);
	}

	/** Default destructor */
	virtual ~CSet()
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
	virtual const char* get_name() const { return "Set"; }

	/** Add an element to the set
	 *
	 * @param e elemet to be added
	 */
	void add(const T& e)
	{
		int32_t index=hash(e);
		if (chain_search(index, e)==NULL)
		{
			insert_key(index, e);
			num_elements++;
		}
	}

	/** Remove an element from the set
	 *
	 * @param e element to be looked for
	 */
	bool contains(const T& e)
	{
		int32_t index=hash(e);
		if (chain_search(index, e)!=NULL)
		{
			return true; 
		}

		return false;
	}

	/** Remove an element from the set
	 *
	 * @param e element to be removed
	 */
	void remove(const T& e)
	{
		int32_t index=hash(e);
		HashSetNode<T>* result=chain_search(index, e);

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
	int32_t index_of(const T& e)
	{
		int32_t index=hash(e);
		HashSetNode<T>* result=chain_search(index, e);

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
		return array->get_element(index)->element;
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
		return &(array->get_element(index)->element);
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
		return array->get_element(index)->element;
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
	int32_t hash(const T& key)
	{
		return CHash::MurmurHash2((uint8_t*)(&key), sizeof(int32_t), 0xDEADBEEF) % hash_size;
	}

	bool is_free(HashSetNode<T>* node)
	{
		if ((node->left==NULL) && (node->right==NULL))
		{
			return true;
		}

		return false;
	}

	/** Searchs key in list(chain) */
	HashSetNode<T>* chain_search(int32_t index, const T& key)
	{
		if (hash_array[index]==NULL)
		{
			return NULL;
		}
		else
		{
			HashSetNode<T>* current=hash_array[index];

			do // iterating all items in the list
			{
				if (current->element==key)
				{
					return current; // it's a search key
				}

				current=current->right;

			} while (current!=NULL);

			return NULL;
		}
	}
	
	/** Inserts nodes with certain key and data in set */
	void insert_key(int32_t index, const T& e)
	{
		int32_t new_index;
		HashSetNode<T>* new_node;

		if(array==NULL)
		{
			return;
		}

		if ((free_index>=array->get_num_elements()) || (array->get_element(free_index)==NULL))
		{
			// init new node
			if(use_sg_mallocs)
			{
				new_node=SG_MALLOC(HashSetNode<T>, 1);
			}
			else
			{
				new_node=(HashSetNode<T>*) calloc(1, sizeof(HashSetNode<T>));
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
		new_node->element=e;
		new_node->left=new_node; // self referencing
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
	void delete_key(int32_t index, HashSetNode<T>* node)
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
		node->left=NULL;
		node->right=NULL;

		free_index=temp;		
	}


protected:
	/** */
	bool use_sg_mallocs;
	/** hashtable size */
	int32_t hash_size;

	/** next free index for new element */
	int32_t free_index;

	/** number of elements */
	int32_t num_elements;

	/** array of lists (chains) */
	HashSetNode<T>** hash_array;

	/** array for index permission */
	DynArray<HashSetNode<T>*>* array;
};

}

#endif /* _SET_H_ */
