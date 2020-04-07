/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Sergey Lisitsyn, Evan Shelhamer, 
 *          Yuyu Zhang
 */

#ifndef __GCARRAY_H__
#define __GCARRAY_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
#define IGNORE_IN_CLASSLIST
/** @brief Template class GCArray implements a garbage collecting static array
 *
 * This array is meant to be used for Shogun Objects (CSGObject) only, as it
 * deals with garbage collection, i.e. on read and array assignment the
 * reference count is increased (and decreased on delete and overwriting
 * elements).
 *
 * */
IGNORE_IN_CLASSLIST template <class T> class CGCArray : public SGObject
{
	public:
		/** default constructor  */
		CGCArray() : SGObject()
		{
			unstable(SOURCE_LOCATION);

			array = NULL;
			size=0;
		}

		/** Constructor
		  *
		  * @param sz length of array
		  */
		CGCArray(int32_t sz) : SGObject()
		{
			ASSERT(sz>0)
			array = SG_CALLOC(T, sz);
			size=sz;
		}

		/** Destructor */
		~CGCArray() override
		{
			for (int32_t i=0; i<size; i++)
				SG_UNREF(array[i]);
			SG_FREE(array);
		}

		/** write access operator
		 *
		 * @param element - element to write
		 * @param index - index to write to
		 */
		inline void set(T element, int32_t index)
		{
			ASSERT(index>=0)
			ASSERT(index<size)
			SG_REF(element);
			SG_UNREF(array[index]);
			array[index]=element;
		}

		/** read only access operator
		 *
		 * @param index index to write to
		 * @return element element
		 */
		inline T get(int32_t index)
		{
			ASSERT(index>=0)
			ASSERT(index<size)
			T element=array[index];
			SG_REF(element); //???
			return element;
		}

		/** get the name of the object
		 *
		 * @return name of object
		 */
		const char* get_name() const override { return "GCArray"; }

	protected:
		/// array
		T* array;
		/// size of array
		int32_t size;
};
}
#endif //__GCARRAY_H__
