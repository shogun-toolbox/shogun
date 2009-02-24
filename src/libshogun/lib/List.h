/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIST_H_
#define _LIST_H_

#include "lib/common.h"
#include "base/SGObject.h"

/** Class ListElement, defines how an element of the the list looks like */
template <class T> class CListElement
{
	public:
		/** next element in list */
		CListElement* next;
		/** previous element in list */
		CListElement* prev;
		/** data of this element */
		T data;

	public:
		/** constructor
		 *
		 * @param p_data data of this element
		 * @param p_prev previous element
		 * @param p_next next element
		 */
		CListElement(T p_data, CListElement* p_prev = NULL, CListElement* p_next = NULL)
		{
			this->data = p_data;
			this->next = p_next;
			this->prev = p_prev;
		};

		/// destructor
		~CListElement() { data = NULL; }
};

/** Class List implements a doubly connected list for low-level-objects. For
 * higher level objects pointers should be used. The list supports calling
 * delete() of an object that is to be removed from the list.
 */
template <class T> class CList : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param p_delete_data if data shall be deleted
		 */
		CList(bool p_delete_data=false) : CSGObject()
		{
			first  = NULL;
			current = NULL;
			last   = NULL;

			num_elements = 0;
			this->delete_data=p_delete_data;
		}

		~CList()
		{
			while (get_num_elements())
			{
				T d=delete_element();

				if (delete_data)
					SG_UNREF(d);
			}
		}

		/** get number of elements in list
		 *
		 * @return number of elements in list
		 */
		inline int32_t get_num_elements() { return num_elements; }

		/** go to first element in list and return it
		 *
		 * @return first element in list or NULL if list is empty
		 */
		inline T get_first_element()
		{
			if (first != NULL)
			{
				current = first;
				if (delete_data)
					SG_REF(current->data);
				return current->data;
			}
			else 
				return NULL;
		}

		/** go to last element in list and return it
		 *
		 * @return last element in list or NULL if list is empty
		 */
		inline T get_last_element()
		{
			if (last != NULL)
			{
				current = last;
				if (delete_data)
					SG_REF(current->data);
				return current->data;
			}
			else 
				return NULL;
		}

		/** go to next element in list and return it
		 *
		 * @return next element in list or NULL if list is empty
		 */
		inline T get_next_element()
		{
			if ((current != NULL) && (current->next != NULL))
			{
				current = current->next;
				if (delete_data)
					SG_REF(current->data);
				return current->data;
			}
			else
				return NULL;
		}

		/** go to previous element in list and return it
		 *
		 * @return previous element in list or NULL if list is empty
		 */
		inline T get_previous_element()
		{
			if ((current != NULL) && (current->prev != NULL))
			{
				current = current->prev;
				if (delete_data)
					SG_REF(current->data);
				return current->data;
			}
			else
				return NULL;
		}

		/** get current element in list
		 *
		 * @return current element in list or NULL if not available
		 */
		inline T get_current_element()
		{
			if (current != NULL)
			{
				if (delete_data)
					SG_REF(current->data);
				return current->data;
			}
			else 
				return NULL;
		}


		/** @name thread safe list access functions */
		//@{

		/** go to first element in list and return it
		 *
		 * @param p_current current list element
		 * @return first element in list or NULL if list is empty
		 */
		inline T get_first_element(CListElement<T> *&p_current)
		{
			if (first != NULL)
			{
				p_current = first;
				if (delete_data)
					SG_REF(p_current->data);
				return p_current->data;
			}
			else
				return NULL;
		}

		/** go to last element in list and return it
		 *
		 * @param p_current current list element
		 * @return last element in list or NULL if list is empty
		 */
		inline T get_last_element(CListElement<T> *&p_current)
		{
			if (last != NULL)
			{
				p_current = last;
				if (delete_data)
					SG_REF(p_current->data);
				return p_current->data;
			}
			else
				return NULL;
		}

		/** go to next element in list and return it
		 *
		 * @param p_current current list element
		 * @return next element in list or NULL if list is empty
		 */
		inline T get_next_element(CListElement<T> *& p_current)
		{
			if ((p_current != NULL) && (p_current->next != NULL))
			{
				p_current = p_current->next;
				if (delete_data)
					SG_REF(p_current->data);
				return p_current->data;
			}
			else
				return NULL;
		}

		/** go to previous element in list and return it
		 *
		 * @param p_current current list element
		 * @return previous element in list or NULL if list is empty
		 */
		inline T get_previous_element(CListElement<T> *& p_current)
		{
			if ((p_current != NULL) && (p_current->prev != NULL))
			{
				p_current = p_current->prev;
				if (delete_data)
					SG_REF(p_current->data);
				return p_current->data;
			}
			else
				return NULL;
		}

		/** get current element in list
		 *
		 * @param p_current current list element
		 * @return current element in list or NULL if not available
		 */
		inline T get_current_element(CListElement<T> *& p_current)
		{
			if (p_current != NULL)
			{
				if (delete_data)
					SG_REF(p_current->data);
				return p_current->data;
			}
			else 
				return NULL;
		}
		//@}

		/** append element AFTER the current element
		 *
		 * @param data data element to append
		 * @return if appending was successful
		 */
		inline bool append_element(T data)
		{
			if (current != NULL)    // none available, case is shattered in insert_element()
			{
				T e=get_next_element();
				if (e)
				{
					if (delete_data)
						SG_UNREF(e);
					// if successor exists use insert_element()
					return insert_element(data);
				}
				else
				{
					// case with no successor but nonempty
					CListElement<T>* element;

					if ((element = new CListElement<T>(data, current)) != NULL)
					{
						current->next = element;
						current       = element;
						last         = element;

						num_elements++;

						if (delete_data)
							SG_REF(data);

						return true;
					}
					else
						return false;
				}
			}
			else
				return insert_element(data);
		}

		/** insert element BEFORE the current element
		 *
		 * @param data data element to insert
		 * @return if inserting was successful
		 */
		inline bool insert_element(T data)
		{
			CListElement<T>* element;

			if (delete_data)
				SG_REF(data);

			if (current == NULL)
			{
				if ((element = new CListElement<T> (data)) != NULL)
				{
					current = element;
					first  = element;
					last   = element;

					num_elements++;

					return true;
				}
				else
					return false;
			}
			else
			{
				if ((element = new CListElement<T>(data, current->prev, current)) != NULL)
				{
					if (current->prev != NULL)
						current->prev->next = element;
					else
						first = element;

					current->prev = element;
					current       = element;

					num_elements++;

					return true;
				}
				else
					return false;
			}
		}

		/** erases current element
		 * the new current element is the successor of the former
		 * current element
		 *
		 * @return the elements data - if available - is returned else NULL
		 */
		inline T delete_element(void)
		{
			T data = get_current_element();

			if (data)
			{
				if (delete_data)
					SG_UNREF(data);

				CListElement<T> *element = current;

				if (element->prev)
					element->prev->next = element->next;

				if (element->next)
					element->next->prev = element->prev;

				if (element->next)
					current = element->next;
				else
					current = element->prev;

				if (element == first)
					first = element->next;

				if (element == last)
					last  = element->prev;

				delete element;

				num_elements--;

				return data;
			} 

			return NULL;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "List"; }

	private:
		/** if data is to be deleted on object destruction */
		bool delete_data;
		/** first element in list */
		CListElement<T>* first;
		/** current element in list */
		CListElement<T>* current;
		/** last element in list */
		CListElement<T>* last;
		/** number of elements */
		int32_t num_elements;
};
#endif
