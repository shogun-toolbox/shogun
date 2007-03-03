/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#include "lib/io.h"
#include "base/Parallel.h"
#include "base/Version.h"

class CSGObject;
class CIO;

class CSGObject
{
public:
	inline CSGObject() : refcount(0) {}

#ifdef HAVE_SWIG
	inline CSGObject(const CSGObject& orig) 
		: refcount(0) , parallel(orig.parallel), io(orig.io) {}
#else
	inline CSGObject(const CSGObject& orig) 
		: refcount(0) {}
#endif

    virtual ~CSGObject()
    {
    }

	inline INT ref()
	{
		return ++refcount;
	}

	inline INT ref_count() const
	{
#ifdef HAVE_SWIG 
		SG_DEBUG("ref_count(): refcount is: %d\n", refcount);
#endif
		return refcount;
	}

	inline INT unref()
	{
#ifdef HAVE_SWIG 
		if (refcount == 0 || --refcount == 0 )
        {
            SG_DEBUG("unref():%ld obj:%x destroying\n", refcount, (ULONG) this);
			//don't do this yet for static interfaces (as none is
			//calling ref/unref properly)
			delete this;
            return 0;
        }
        else
        {
            SG_DEBUG("unref():%ld obj:%x decreased\n", refcount, (ULONG) this);
            return refcount;
        }
#else
		if (refcount>0)
			return --refcount;
		else
			return 0;
#endif
	}

private:
	INT refcount;

public:
#ifdef HAVE_SWIG
	CParallel parallel;
	CIO io;
	CVersion version;
#else
	static CParallel parallel;
	static CIO io;
	static CVersion version;
#endif
};
#endif // __SGOBJECT_H__
