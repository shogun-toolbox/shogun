#ifndef _LIST_H_
#define _LIST_H_

#include "lib/common.h"

/// doubly connected list for low-level-objects. use pointers to higher-level objects
template <class T> class CList
{
	class CListElement
	{
		public:
			CListElement* next;
			CListElement* prev;
			T data;

		public:
			CListElement(T data, CListElement* prev = NULL, CListElement* next = NULL)
			{
				this->data = data;
				this->next = next;
				this->prev = prev;
			}

			/// Destruktor
			~CListElement()
			{
				data = NULL;
			}
	};

public:
	CList(bool delete_data=false)
	{
		first  = NULL;
		actual = NULL;
		last   = NULL;

		num_elements = 0;
		this->delete_data=delete_data;
	}

	~CList()
	{
		while (get_num_elements())
		{
			T d=delete_element();
			if (delete_data)
				delete d;
		}
	}

	/// number of elements in list
	inline int get_num_elements()
	{
		return num_elements;
	}

	/// go to first element in list and return it (or NULL if list is empty)
	inline T get_first_element()
	{
		if (first != NULL)
		{
			actual = first;
			return actual->data;
		}
		else 
			return NULL;
	}

	/// go to last element in list and return it (or NULL if list is empty)
	inline T get_last_element()
	{
		if (last != NULL)
		{
			actual = last;
			return actual->data;
		}
		else 
			return NULL;
	}

	/// go to next element in list and return it (or NULL if not available)
	inline T get_next_element()
	{
		if ((actual != NULL) && (actual->next != NULL))
		{
			actual = actual->next;
			return actual->data;
		}
		else
			return NULL;
	}

	/// go to previous element in list and return it (or NULL if not available)
	inline T get_previous_element()
	{
		if ((actual != NULL) && (actual->prev != NULL))
		{
			actual = actual->prev;
			return actual->data;
		}
		else
			return NULL;
	}

	/// return current element in list (or NULL if not available)
	inline T get_current_element()
	{
		if (actual != NULL)
			return actual->data;
		else 
			return NULL;
	}

	/// append element AFTER the current element. return true on success
	inline bool append_element(T data)
	{
		if (actual != NULL)    // none available, case is shattered in insert_element()
		{
			if (get_next_element())
			{
				// if successor exists use insert_element()
				return insert_element(data);    
			}
			else
			{
				// case with no successor but nonempty
				CListElement* element;

				if ((element = new CListElement(data, actual)) != NULL)
				{
					actual->next = element;
					actual       = element;
					last         = element;

					num_elements++;

					return true;
				}
				else
					return false;
			}
		}
		else 
			return insert_element(data);   
	}

	/// insert element BEFORE the current element. return true on success
	inline bool insert_element(T data)
	{
		CListElement* element;

		if (actual == NULL)                 
		{
			if ((element = new CListElement (data)) != NULL)
			{
				actual = element;
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
			if ((element = new CListElement(data, actual->prev, actual)) != NULL)
			{
				if (actual->prev != NULL)
					actual->prev->next = element;
				else
					first = element;

				actual->prev = element;
				actual       = element;

				num_elements++;

				return true;
			}
			else
				return false;
		}
	}

	/** erases current element; the new current element is the successor of the former
	 * current element. the elements data - if available - is returned
	 * else NULL */
	inline T delete_element(void)
	{
		T data = get_current_element();

		if (data)
		{
			CListElement *element = actual;

			if (element->prev)
				element->prev->next = element->next;

			if (element->next)
				element->next->prev = element->prev; 

			if (element->next)
				actual = element->next;
			else
				actual = element->prev;

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

private:
	bool delete_data;
	CListElement* first;
	CListElement* actual;
	CListElement* last;
	int num_elements;
};
#endif
