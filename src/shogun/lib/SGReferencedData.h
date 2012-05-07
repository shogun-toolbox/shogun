/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */
#ifndef __SGREFERENCED_DATA_H__
#define __SGREFERENCED_DATA_H__

#include <shogun/io/SGIO.h>

namespace shogun
{
/** @brief shogun reference count managed data */
class SGReferencedData
{
	public:
		/** default constructor */
		SGReferencedData(bool ref_counting=true) : m_refcount(NULL)
		{ 
			if (ref_counting)
				m_refcount=SG_CALLOC(int32_t, 1); 

			ref();
		}

		/** copy constructor */
		SGReferencedData(const SGReferencedData &orig)
			: m_refcount(orig.m_refcount) 
		{
			ref();
		}

		/** override assignment operator to increase refcount on assignments */
		SGReferencedData& operator= (const SGReferencedData &orig)
		{
			if (this == &orig)
				return *this;

			unref();
			copy_data(orig);
			copy_refcount(orig);
			ref();
			return *this;
		}

		/** empty destructor
		 *
		 * NOTE: unref() has to be called in derived classes
		 * to avoid memory leaks.
		 */
		virtual ~SGReferencedData()
		{
		}

#ifdef USE_REFERENCE_COUNTING
		/** increase reference counter
		 *
		 * @return reference count
		 */
		int32_t ref()
		{
			if (m_refcount == NULL)
			{
				return -1;
			}

			++(*m_refcount);
#ifdef DEBUG_SGVECTOR
			SG_SGCDEBUG("ref() refcount %ld data %p increased\n", *m_refcount, this);
#endif
			return *m_refcount;
		}

		/** display reference counter
		 *
		 * @return reference count
		 */
		int32_t ref_count()
		{
			if (m_refcount == NULL)
				return -1;

#ifdef DEBUG_SGVECTOR
			SG_SGCDEBUG("ref_count(): refcount %d, data %p\n", *m_refcount, this);
#endif
			return *m_refcount;
		}

		/** decrement reference counter and deallocate object if refcount is zero
		 * before or after decrementing it
		 *
		 * @return reference count
		 */
		int32_t unref()
		{
			if (m_refcount == NULL)
			{
				init_data();
				return -1;
			}

			if (*m_refcount==0 || --(*m_refcount)==0)
			{
#ifdef DEBUG_SGVECTOR
				SG_SGCDEBUG("unref() refcount %d data %p destroying\n", *m_refcount, this);
#endif
				free_data();
				SG_FREE(m_refcount);
				m_refcount=NULL;
				return 0;
			}
			else
			{
#ifdef DEBUG_SGVECTOR
				SG_SGCDEBUG("unref() refcount %d data %p decreased\n", *m_refcount, this);
#endif
				init_data();
				return *m_refcount;
			}
		}

#endif //USE_REFERENCE_COUNTING

	protected:
		void copy_refcount(const SGReferencedData &orig)
		{
			m_refcount=orig.m_refcount;
		}

		/** needs to be overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig)=0;

		/** needs to be overridden to initialize empty data */
		virtual void init_data()=0;

		/** needs to be overridden to free data */
		virtual void free_data()=0;

	private:
		/** reference counter */
		int32_t* m_refcount;
};
}
#endif // __SGREFERENCED_DATA_H__
