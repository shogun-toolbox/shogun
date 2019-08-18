#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/RefCount.h>

using namespace shogun;

namespace shogun {

SGReferencedData::SGReferencedData(bool ref_counting) : m_refcount(NULL)
{
	if (ref_counting)
	{
		m_refcount = new RefCount(0);
	}

	ref();
}

SGReferencedData::SGReferencedData(const SGReferencedData &orig)
{
	copy_refcount(orig);
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
	delete m_refcount;
}

int32_t SGReferencedData::ref_count()
{
	if (m_refcount == NULL)
		return -1;

	int32_t c = m_refcount->ref_count();

#ifdef DEBUG_SGVECTOR
	SG_GCDEBUG("ref_count(): refcount {}, data {}", c, fmt::ptr(this))
#endif
	return c;
}

/** copy refcount */
void SGReferencedData::copy_refcount(const SGReferencedData &orig)
{
	m_refcount =  orig.m_refcount;
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

	int32_t c = m_refcount->ref();

#ifdef DEBUG_SGVECTOR
	SG_GCDEBUG("ref() refcount {} data {} increased", c, fmt::ptr(this))
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

	int32_t c = m_refcount->unref();

	if (c<=0)
	{
#ifdef DEBUG_SGVECTOR
		SG_GCDEBUG("unref() refcount {} data {} destroying", c, fmt::ptr(this))
#endif
		free_data();
		delete m_refcount;
		m_refcount=NULL;
		return 0;
	}
	else
	{
#ifdef DEBUG_SGVECTOR
		SG_GCDEBUG("unref() refcount {} data {} decreased", c, fmt::ptr(this))
#endif
		init_data();
		m_refcount=NULL;
		return c;
	}
}
}
