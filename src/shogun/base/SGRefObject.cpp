/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2013 Thoralf Klein
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include <base/init.h>
#include <base/SGRefObject.h>
#include <io/SGIO.h>

#include <stdlib.h>
#include <stdio.h>

using namespace shogun;

SGRefObject::SGRefObject()
{
	init();
	m_refcount = new RefCount(0);

	SG_SGCDEBUG("SGRefObject created (%p)\n", this)
}

SGRefObject::SGRefObject(const SGRefObject& orig)
{
	init();
	m_refcount = orig.m_refcount;
	SG_REF(this);
}

SGRefObject::~SGRefObject()
{
	SG_SGCDEBUG("SGRefObject destroyed (%p)\n", this)
	delete m_refcount;
}

#ifdef USE_REFERENCE_COUNTING
int32_t SGRefObject::ref()
{
	int32_t count = m_refcount->ref();
	SG_SGCDEBUG("ref() refcount %ld obj %s (%p) increased\n", count, this->get_name(), this)
	return m_refcount->ref_count();
}

int32_t SGRefObject::ref_count()
{
	int32_t count = m_refcount->ref_count();
	SG_SGCDEBUG("ref_count(): refcount %d, obj %s (%p)\n", count, this->get_name(), this)
	return m_refcount->ref_count();
}

int32_t SGRefObject::unref()
{
	int32_t count = m_refcount->unref();
	if (count<=0)
	{
		SG_SGCDEBUG("unref() refcount %ld, obj %s (%p) destroying\n", count, this->get_name(), this)
		delete this;
		return 0;
	}
	else
	{
		SG_SGCDEBUG("unref() refcount %ld obj %s (%p) decreased\n", count, this->get_name(), this)
		return m_refcount->ref_count();
	}
}
#endif //USE_REFERENCE_COUNTING

#ifdef TRACE_MEMORY_ALLOCS
#include <lib/Map.h>
extern CMap<void*, shogun::MemoryBlock>* sg_mallocs;
#endif

void SGRefObject::init()
{
#ifdef TRACE_MEMORY_ALLOCS
	if (sg_mallocs)
	{
		int32_t idx=sg_mallocs->index_of(this);
		if (idx>-1)
		{
			MemoryBlock* b=sg_mallocs->get_element_ptr(idx);
			b->set_sgobject();
		}
	}
#endif
}
