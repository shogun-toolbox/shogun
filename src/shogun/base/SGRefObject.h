/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2010 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Written (W) 2013 Thoralf Klein
 * Copyright (C) 2008-2010 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __SGREFOBJECT_H__
#define __SGREFOBJECT_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/RefCount.h>

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{

// define reference counter macros
//
#ifdef USE_REFERENCE_COUNTING
#define SG_REF(x) { if (x) (x)->ref(); }
#define SG_UNREF(x) { if (x) { if ((x)->unref()==0) (x)=NULL; } }
#define SG_UNREF_NO_NULL(x) { if (x) { (x)->unref(); } }
#else
#define SG_REF(x)
#define SG_UNREF(x)
#define SG_UNREF_NO_NULL(x)
#endif

/** @brief Class SGObject is the base class of all shogun objects.
 *
 * Apart from dealing with reference counting that is used to manage shogung
 * objects in memory (erase unused object, avoid cleaning objects when they are
 * still in use), it provides interfaces for:
 *
 * -# parallel - to determine the number of used CPUs for a method (cf. Parallel)
 * -# io - to output messages and general i/o (cf. IO)
 * -# version - to provide version information of the shogun version used (cf. Version)
 *
 * All objects can be cloned and compared (deep copy, recursively)
 */
class SGRefObject
{
public:
	/** default constructor */
	SGRefObject();

	/** copy constructor */
	SGRefObject(const SGRefObject& orig);

	/** destructor */
	virtual ~SGRefObject();

#ifdef USE_REFERENCE_COUNTING
	/** increase reference counter
	 *
	 * @return reference count
	 */
	int32_t ref();

	/** display reference counter
	 *
	 * @return reference count
	 */
	int32_t ref_count();

	/** decrement reference counter and deallocate object if refcount is zero
	 * before or after decrementing it
	 *
	 * @return reference count
	 */
	int32_t unref();
#endif //USE_REFERENCE_COUNTING

	/** A shallow copy.
	 * All the SGObject instance variables will be simply assigned and SG_REF-ed.
	 */
	virtual SGRefObject *shallow_copy() const
	{
		SG_SNOTIMPLEMENTED
		return NULL;
	}

	/** A deep copy.
	 * All the instance variables will also be copied.
	 */
	virtual SGRefObject *deep_copy() const
	{
		SG_SNOTIMPLEMENTED
		return NULL;
	}

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name() const = 0;

#ifdef TRACE_MEMORY_ALLOCS
	static void list_memory_allocs()
	{
		shogun::list_memory_allocs();
	}
#endif

private:
	void init();

private:

	RefCount* m_refcount;
};
}
#endif // __SGREFOBJECT_H__
