#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

SGReferencedData::SGReferencedData(bool ref_counting) : m_refcount(NULL)
{ 
	if (ref_counting)
	{
		m_refcount = SG_CALLOC(refcount_t, 1);
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK_INIT(&m_refcount->lock);
#endif
	}

	ref();
}

SGReferencedData::SGReferencedData(const SGReferencedData &orig)
	: m_refcount(orig.m_refcount)
{
	ref();
}

SGReferencedData& SGReferencedData::operator= (const SGReferencedData &orig)
{
	if (this == &orig)
		return *this;

	unref();
	copy_data(orig);
	copy_refcount(orig);
	ref();
	return *this;
}

SGReferencedData::~SGReferencedData()
{
}

int32_t SGReferencedData::ref_count()
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

/** copy refcount */
void SGReferencedData::copy_refcount(const SGReferencedData &orig)
{
	m_refcount=orig.m_refcount;
}

/** increase reference counter
 *
 * @return reference count
 */
int32_t SGReferencedData::ref()
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
int32_t SGReferencedData::unref()
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
