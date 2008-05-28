/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include "lib/io.h"
#include "base/Parallel.h"
#include "base/Version.h"

class CSGObject;
class CIO;

// define reference counter macros
#if defined(HAVE_SWIG)
#define SG_REF(x) { if (x) x->ref(); }
#define SG_UNREF(x) { if (x) x->unref(); } 
#else
#define SG_REF(x)
#define SG_UNREF(x)
#endif // HAVE_SWIG
class CSGObject
{
public:
    virtual ~CSGObject()
    {
    }

#ifdef HAVE_SWIG
	inline CSGObject() : refcount(0)
	{
	}

	inline CSGObject(const CSGObject& orig)
		: refcount(0) , parallel(orig.parallel), io(orig.io)
	{
	}

	inline INT ref()
	{
		++refcount;
		SG_DEBUG("ref():%ld obj:%p\n", refcount, this);
		return refcount;
	}

	inline INT ref_count() const
	{
		SG_DEBUG("ref_count(): refcount is: %d\n", refcount);
		return refcount;
	}

	inline INT unref()
	{
		if (refcount==0 || --refcount==0)
		{
			SG_DEBUG("unref():%ld obj:%p destroying\n", refcount, this);
			//don't do this yet for static interfaces (as none is
			//calling ref/unref properly)
			delete this;
			return 0;
		}
		else
		{
			SG_DEBUG("unref():%ld obj:%p decreased\n", refcount, this);
			return refcount;
		}
	}

private:
	INT refcount;

public:
	CParallel parallel;
	CIO io;
	CVersion version;
#else // HAVE_SWIG
public:
	inline CSGObject()
	{
	}

	inline CSGObject(const CSGObject& orig)
	{
	}

public:
	static CParallel parallel;
	static CIO io;
	static CVersion version;
#endif // HAVE_SWIG
};
#endif // __SGOBJECT_H__
