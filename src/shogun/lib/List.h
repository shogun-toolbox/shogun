/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LIST_H_
#define _LIST_H_

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
/** @brief Class ListElement, defines how an element of the the list looks like */
class CListElement :public CSGObject
{
	public:
		/** default constructor */
		CListElement()
			: next(NULL), prev(NULL), data(NULL)
		{
			init();
		}

		/** constructor
		 *
		 * @param p_data data of this element
		 * @param p_prev previous element
		 * @param p_next next element
		 */
		CListElement(CSGObject* p_data,
				CListElement* p_prev = NULL,
				CListElement* p_next = NULL)
		{
			init();

			this->data = p_data;
			this->next = p_next;
			this->prev = p_prev;
		}

		/// destructor
		virtual ~CListElement() { data = NULL; }

		/** @return object name */
		virtual const char* get_name() const { return "ListElement"; }

	private:
		void init()
		{
			m_parameters->add(&data, "data", "Data of this element.");
			m_parameters->add((CSGObject**) &next, "next",
					"Next element in list.");
			m_model_selection_parameters->add((CSGObject**) &next, "next",
					"Next element in list.");
			m_model_selection_parameters->add(&data, "data", "Data of this element.");
		}

	public:
		/** next element in list */
		CListElement* next;
		/** previous element in list */
		CListElement* prev;
		/** data of this element */
		CSGObject* data;

};

/** @brief Class List implements a doubly connected list for low-level-objects.
 *
 * For higher level objects pointers should be used. The list supports calling
 * delete() of an object that is to be removed from the list.
 */
class CList : public CSGObject
{
	public:
		/** constructor
		 *
		 * @param p_delete_data if data shall be deleted
		 */
		CList(bool p_delete_data=false) : CSGObject()
		{
			m_parameters->add(&delete_data, "delete_data",
							  "Delete data on destruction?");
			m_parameters->add(&num_elements, "num_elements",
							  "Number of elements.");
			m_parameters->add((CSGObject**) &first, "first",
							  "First element in list.");
			m_model_selection_parameters->add((CSGObject**) &first, "first",
								  "First element in list.");

			first  = NULL;
			current = NULL;
			last   = NULL;

			num_elements = 0;
			this->delete_data=p_delete_data;
		}

		virtual ~CList()
		{
			SG_DEBUG("Destroying List %p\n", this)

			delete_all_elements();
		}

		/** deletes all elements from list */
		inline void delete_all_elements()
		{
			while (get_num_elements())
			{
				CSGObject* d=delete_element();

				if (delete_data)
				{
					SG_DEBUG("SG_UNREF List Element %p\n", d)
					SG_UNREF(d);
				}
			}

			first=NULL;
			current=NULL;
			last=NULL;
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
		inline CSGObject* get_first_element()
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
		inline CSGObject* get_last_element()
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
		inline CSGObject* get_next_element()
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
		inline CSGObject* get_previous_element()
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
		inline CSGObject* get_current_element()
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
		inline CSGObject* get_first_element(CListElement*& p_current)
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
		inline CSGObject* get_last_element(CListElement*& p_current)
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
		inline CSGObject* get_next_element(CListElement*& p_current)
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
		inline CSGObject* get_previous_element(CListElement*& p_current)
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
		inline CSGObject* get_current_element(CListElement*& p_current)
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
		inline bool append_element(CSGObject* data)
		{
			if (current != NULL)    // none available, case is shattered in insert_element()
			{
				CSGObject* e=get_next_element();
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
					CListElement* element;

					if ((element = new CListElement(data, current)) != NULL)
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

		/** append at end of list
		 *
		 * @param data data element to append
		 * @return if appending was successful
		 */
		inline bool append_element_at_listend(CSGObject* data)
		{
			CSGObject* p = get_last_element();
			if (delete_data)
				SG_UNREF(p);

			return append_element(data);
		}

		/** append at end of list
		 *
		 * @param data data element to append
		 * @return if appending was successful
		 */
		inline bool push(CSGObject* data)
		{
			return append_element_at_listend(data);
		}

		/** removes last element of list
		 *
		 * @return if deletion was successful
		 */
		inline bool pop()
		{
			if (last)
			{
				if (first==last)
					first=NULL;

				if (current==last)
				{
					if (first==last)
						current=NULL;
					else
						current=current->prev;
				}

				if (delete_data)
					SG_UNREF(last->data);

				CListElement* temp=last;
				last=last->prev;
				SG_UNREF(temp);
				if (last)
					last->next=NULL;

				num_elements--;

				return true;
			}
			else
				return false;
		}

		/** insert element BEFORE the current element
		 *
		 * @param data data element to insert
		 * @return if inserting was successful
		 */
		inline bool insert_element(CSGObject* data)
		{
			CListElement* element;

			if (delete_data)
				SG_REF(data);

			if (current == NULL)
			{
				if ((element = new CListElement(data)) != NULL)
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
				if ((element = new CListElement(data, current->prev, current)) != NULL)
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
		 * @return the elements data - if available - otherwise NULL
		 */
		inline CSGObject* delete_element()
		{
			CSGObject* data = get_current_element();

			if (num_elements>0)
				num_elements--;

			if (data)
			{
				if (delete_data)
					SG_UNREF(data);

				CListElement *element = current;

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

				return data;
			}

			return NULL;
		}

		virtual void load_serializable_post() throw (ShogunException)
		{
			CSGObject::load_serializable_post();

			current = first;
			CListElement* prev = NULL;
			for (CListElement* cur=first; cur!=NULL; cur=cur->next)
			{
				cur->prev = prev;
				prev = cur;
			}
			last = prev;
		}

		/** print all elements of the list */
		void print_list()
		{
			CListElement* c=first;

			while (c)
			{
				SG_PRINT("\"%s\" at %p\n", c->data ? c->data->get_name() : "", c->data)
				c=c->next;
			}
		}

		/** @return delete_data flag which indicates if list SG_REF's stuff */
		inline bool get_delete_data() { return delete_data; }

		/** @return object name */
		virtual const char* get_name() const { return "List"; }

	private:
		/** if data is to be deleted on object destruction */
		bool delete_data;
		/** first element in list */
		CListElement* first;
		/** current element in list */
		CListElement* current;
		/** last element in list */
		CListElement* last;
		/** number of elements */
		int32_t num_elements;
};
}
#endif
