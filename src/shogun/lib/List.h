/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Jacob Walker, Yuyu Zhang,
 *          Evan Shelhamer, Soumyajit De
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
class ListElement :public SGObject
{
	public:
		/** default constructor */
		ListElement()
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
		ListElement(std::shared_ptr<SGObject> p_data,
				std::shared_ptr<ListElement> p_prev = NULL,
				std::shared_ptr<ListElement> p_next = NULL)
		{
			init();

			this->data = p_data;
			this->next = p_next;
			this->prev = p_prev;
		}

		/// destructor
		virtual ~ListElement() { data = NULL; }

		/** @return object name */
		virtual const char* get_name() const { return "ListElement"; }

	private:
		void init()
		{
			/*SG_ADD(&data, "data", "Data of this element.")*/;
			/*SG_ADD(&next, "next", "Next element in list.")*/;
		}

	public:
		/** next element in list */
		std::shared_ptr<ListElement> next;
		/** previous element in list */
		std::shared_ptr<ListElement> prev;
		/** data of this element */
		std::shared_ptr<SGObject> data;

};

/** @brief Class List implements a doubly connected list for low-level-objects.
 *
 * For higher level objects pointers should be used. The list supports calling
 * delete() of an object that is to be removed from the list.
 */
class List : public SGObject
{
	public:
		/** constructor
		 *
		 * @param p_delete_data if data shall be deleted
		 */
		List(bool p_delete_data=false) : SGObject()
		{
			/*m_parameters->add(&delete_data, "delete_data",
							  "Delete data on destruction?")*/;
			/*m_parameters->add(&num_elements, "num_elements",
							  "Number of elements.")*/;
			/*m_parameters->add((SGObject**) &first, "first",
							  "First element in list.")*/;
			/*m_model_selection_parameters->add((SGObject**) &first, "first",
								  "First element in list.")*/;

			first  = NULL;
			current = NULL;
			last   = NULL;

			num_elements = 0;
			this->delete_data=p_delete_data;
		}

		virtual ~List()
		{
			SG_DEBUG("Destroying List %p\n", this)

			delete_all_elements();
		}

		/** deletes all elements from list */
		inline void delete_all_elements()
		{
			// move to the first element and then delete sequentially
			auto d=get_first_element();

			while (get_num_elements())
			{
				d=delete_element();

				// we don't need to check for delete_data flag here
				// delete_element() takes care of whether or not
				// data should be SG_UNREF'ed
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
		inline std::shared_ptr<SGObject> get_first_element()
		{
			if (first != NULL)
			{
				current = first;
				if (delete_data)

				return current->data;
			}
			return NULL;
		}

		/** go to last element in list and return it
		 *
		 * @return last element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_last_element()
		{
			if (last != NULL)
			{
				current = last;
				return current->data;
			}
			return NULL;
		}

		/** go to next element in list and return it
		 *
		 * @return next element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_next_element()
		{
			if ((current != NULL) && (current->next != NULL))
			{
				current = current->next;
				return current->data;
			}
			return NULL;
		}

		/** go to previous element in list and return it
		 *
		 * @return previous element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_previous_element()
		{
			if ((current != NULL) && (current->prev != NULL))
			{
				current = current->prev;
				return current->data;
			}
			return NULL;
		}

		/** get current element in list
		 *
		 * @return current element in list or NULL if not available
		 */
		inline std::shared_ptr<SGObject> get_current_element()
		{
			if (current != NULL)
			{
				return current->data;
			}
			return NULL;
		}


		/** @name thread safe list access functions */
		//@{

		/** go to first element in list and return it
		 *
		 * @param p_current current list element
		 * @return first element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_first_element(std::shared_ptr<ListElement>& p_current)
		{
			if (first != NULL)
			{
				p_current = first;
				return p_current->data;
			}
			return NULL;
		}

		/** go to last element in list and return it
		 *
		 * @param p_current current list element
		 * @return last element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_last_element(std::shared_ptr<ListElement>& p_current)
		{
			if (last != NULL)
			{
				p_current = last;
				return p_current->data;
			}
			return NULL;
		}

		/** go to next element in list and return it
		 *
		 * @param p_current current list element
		 * @return next element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_next_element(std::shared_ptr<ListElement>& p_current)
		{
			if ((p_current != NULL) && (p_current->next != NULL))
			{
				p_current = p_current->next;
				return p_current->data;
			}
			return NULL;
		}

		/** go to previous element in list and return it
		 *
		 * @param p_current current list element
		 * @return previous element in list or NULL if list is empty
		 */
		inline std::shared_ptr<SGObject> get_previous_element(std::shared_ptr<ListElement>& p_current)
		{
			if ((p_current != NULL) && (p_current->prev != NULL))
			{
				p_current = p_current->prev;
				return p_current->data;
			}
			return NULL;
		}

		/** get current element in list
		 *
		 * @param p_current current list element
		 * @return current element in list or NULL if not available
		 */
		inline std::shared_ptr<SGObject> get_current_element(std::shared_ptr<ListElement>& p_current)
		{
			if (p_current != NULL)
			{
				return p_current->data;
			}
			return NULL;
		}
		//@}

		/** append element AFTER the current element and move to the newly
		 * added element
		 *
		 * @param data data element to append
		 * @return if appending was successful
		 */
		inline bool append_element(std::shared_ptr<SGObject> data)
		{
			SG_DEBUG("Entering\n");

			// none available, case is shattered in insert_element()
			if (current != NULL)
			{
				auto e=get_next_element();
				if (e)
				{
					// if successor exists use insert_element()
					SG_DEBUG("Leaving\n");
					return insert_element(data);
				}
				else
				{
					// case with no successor but nonempty
					std::shared_ptr<ListElement> element;

					if ((element = std::make_shared<ListElement>(data, current)) != NULL)
					{
						current->next = element;
						current       = element;
						last         = element;

						num_elements++;

						SG_DEBUG("Leaving\n");
						return true;
					}
					else
					{
						SG_WARNING("Error in allocating memory for new element!\n");
						SG_DEBUG("Leaving\n");
						return false;
					}
				}
			}
			else
			{
				SG_DEBUG("Leaving\n");
				return insert_element(data);
			}
		}

		/** append at end of list
		 *
		 * @param data data element to append
		 * @return if appending was successful
		 */
		inline bool append_element_at_listend(std::shared_ptr<SGObject> data)
		{
			auto p = get_last_element();
			return append_element(data);
		}

		/** append at end of list
		 *
		 * @param data data element to append
		 * @return if appending was successful
		 */
		inline bool push(std::shared_ptr<SGObject> data)
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


				auto temp=last;
				last=last->prev;

				if (last)
					last->next=NULL;

				num_elements--;

				return true;
			}
			else
				return false;
		}

		/** insert element BEFORE the current element and move to the newly
		 * added element
		 *
		 * @param data data element to insert
		 * @return if inserting was successful
		 */
		inline bool insert_element(std::shared_ptr<SGObject> data)
		{
			std::shared_ptr<ListElement> element;

			if (current == NULL)
			{
				if ((element = std::make_shared<ListElement>(data)) != NULL)
				{
					current = element;
					first  = element;
					last   = element;

					num_elements++;

					return true;
				}
				else
				{
					SG_WARNING("Error in allocating memory for new element!\n");
					return false;
				}
			}
			else
			{
				if ((element = std::make_shared<ListElement>(data, current->prev, current)) != NULL)
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
				{
					SG_WARNING("Error in allocating memory for new element!\n");
					return false;
				}
			}
		}

		/** erases current element
		 * the new current element is the successor of the former
		 * current element
		 *
		 * @return the elements data - if available - otherwise NULL
		 */
		inline std::shared_ptr<SGObject> delete_element()
		{
			SG_DEBUG("Entering\n");
			auto data = current ? current->data : NULL;

			if (num_elements>0)
				num_elements--;

			if (data)
			{
				auto element = current;

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

				element.reset();

				SG_DEBUG("Leaving\n");
				return data;
			}

			SG_DEBUG("Leaving\n");
			return NULL;
		}

		virtual void load_serializable_post() noexcept(false)
		{
			SGObject::load_serializable_post();

			current = first;
			std::shared_ptr<ListElement> prev = NULL;
			for (auto cur=first; cur!=NULL; cur=cur->next)
			{
				cur->prev = prev;
				prev = cur;
			}
			last = prev;
		}

		/** print all elements of the list */
		void print_list()
		{
			auto c=first;

			while (c)
			{
				SG_PRINT("\"%s\" at %p\n", c->data ? c->data->get_name() : "", c->data.get())
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
		std::shared_ptr<ListElement> first;
		/** current element in list */
		std::shared_ptr<ListElement> current;
		/** last element in list */
		std::shared_ptr<ListElement> last;
		/** number of elements */
		int32_t num_elements;
};
}
#endif
