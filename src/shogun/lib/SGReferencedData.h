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
#include <shogun/base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

/** refcount structure */
struct refcount_t 
{
	/** reference count */
	int32_t rc;
#ifdef HAVE_PTHREAD
	/** lock for thread safety */
	PTHREAD_LOCK_T lock;
#endif
};

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
			{
				m_refcount = SG_CALLOC(refcount_t, 1);
				PTHREAD_LOCK_INIT(&m_refcount->lock);
			}

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

		/** display reference counter
		 *
		 * @return reference count
		 */
		int32_t ref_count()
		{
			if (m_refcount == NULL)
				return -1;

#ifdef HAVE_PTHREAD
			PTHREAD_LOCK(&m_refcount->lock);
#endif
			int32_t c = m_refcount->rc;
#ifdef HAVE_PTHREAD
			PTHREAD_UNLOCK(&m_refcount->lock);
#endif 

#ifdef DEBUG_SGVECTOR
			SG_SGCDEBUG("ref_count(): refcount %d, data %p\n", c, this);
#endif
			return c;
		}

	protected:
		/** copy refcount */
		void copy_refcount(const SGReferencedData &orig)
		{
			m_refcount=orig.m_refcount;
		}

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

#ifdef HAVE_PTHREAD
			PTHREAD_LOCK(&m_refcount->lock);
#endif
			int32_t c = ++(m_refcount->rc);
#ifdef HAVE_PTHREAD
			PTHREAD_UNLOCK(&m_refcount->lock);
#endif 
#ifdef DEBUG_SGVECTOR
			SG_SGCDEBUG("ref() refcount %ld data %p increased\n", c, this);
#endif
			return c;
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
				m_refcount=NULL;
				return -1;
			}

#ifdef HAVE_PTHREAD
			PTHREAD_LOCK(&m_refcount->lock);
#endif
			int32_t c = --(m_refcount->rc);
#ifdef HAVE_PTHREAD
			PTHREAD_UNLOCK(&m_refcount->lock);
#endif 
			if (c<=0)
			{
#ifdef DEBUG_SGVECTOR
				SG_SGCDEBUG("unref() refcount %d data %p destroying\n", c, this);
#endif
				free_data();
#ifdef HAVE_PTHREAD
			PTHREAD_LOCK_DESTROY(&m_refcount->lock);
#endif
				SG_FREE(m_refcount);
				m_refcount=NULL;
				return 0;
			}
			else
			{
#ifdef DEBUG_SGVECTOR
				SG_SGCDEBUG("unref() refcount %d data %p decreased\n", c, this);
#endif
				init_data();
				m_refcount=NULL;
				return c;
			}
		}

		/** needs to be overridden to copy data */
		virtual void copy_data(const SGReferencedData &orig)=0;

		/** needs to be overridden to initialize empty data */
		virtual void init_data()=0;

		/** needs to be overridden to free data */
		virtual void free_data()=0;

	private:

		/** reference counter */
		refcount_t* m_refcount;
};
}
#endif // __SGREFERENCED_DATA_H__
