#ifndef _DYNARRAY_H_
#define _DYNARRAY_H_

#include "lib/common.h"
#include "lib/Mathmatics.h"

/** dynamic array, i.e. array that can be used like a list or an array.
it grows and shrinks dynamically, while elements can be accessed via index
performance tuned for simple types as float etc.
for hi-level objects only store pointers
*/
template <class T> class CDynamicArray
{
public:
	CDynamicArray(int resize_granularity = 128)
	{
		this->resize_granularity = resize_granularity;

		array = (T*) calloc(resize_granularity, sizeof(T));
		assert(array);

		num_elements = resize_granularity;
		last_element_idx = -1;
	}

	~CDynamicArray()
	{
		free(array);
	}

	/// return total array size (including granularity buffer)
	inline int get_num_elements()
	{
		return num_elements;
	}

	/// return index of element which is at the end of the array
	inline int get_last_element_idx()
	{
		return last_element_idx;
	}

	///return array element at index
	inline T get_element(int index)
	{
		assert((array != NULL) && (index >= 0) && (index <= last_element_idx));
		return array[index];
	}

	///set array element at index 'index' return false in case of trouble
	inline bool set_element(T element, int index)
	{
		assert((array != NULL) && (index >= 0));
		if (index <= last_element_idx)
		{
			array[index]=element;
			return true;
		}
		else if (index < num_elements)
		{
			array[index]=element;
			last_element_idx=index;
			return true;
		}
		else
		{
			if (resize_array(index))
				return set_element(element, index);
			else
				return false;
		}
	}

	///set array element at index 'index' return false in case of trouble
	inline bool add_element(T element)
	{
		return set_element(element, last_element_idx+1);
	}

	///set array element at index 'index' return false in case of trouble
	inline bool delete_element(int idx)
	{
		if (idx>=0 && idx<=last_element_idx)
		{
			for (int i=idx; i<last_element_idx; i++)
				array[i]=array[i+1];

			last_element_idx--;

			if ( num_elements - last_element_idx >= resize_granularity)
				resize_array(last_element_idx);
		}
		else
			return false;
	}
	
	bool resize_array(int new_num_elements)
	{
		T* p= (T*) realloc(array, ((new_num_elements/granularity)+1)*resize_granularity);
		if (p)
		{
			num_elements=new_num_elements;
			array=p;
			memset(array[CMath::min(new_num_elements, num_elements)], 0, CMath::abs(new_num_elements-num_elements), sizeof(T));
			return true;
		}
		else
			return false;
	}

	/// operator overload for array read access
	T operator [](int index)
	{
		return get_element(index);
	}
	
	///TODO write access via []
	/// operator overload for array write access
	void operator [](const T value, int index)
	{
		return set_element(value, index);
	}

protected:
	/// shrink/grow step size
	int resize_granularity;

	/// memory for dynamic array
	T **array;

	/// the number of potentially used elements in array
	int num_elements;

	/// the element in the array that has largest index
	int last_element_idx;
};
#endif
