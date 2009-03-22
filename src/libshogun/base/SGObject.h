/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#ifndef __SGOBJECT_H__
#define __SGOBJECT_H__

#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/access.hpp>
#include <string>
#endif //HAVE_BOOST_SERIALIZATION

#include "lib/io.h"
#include "base/Parallel.h"
#include "base/Version.h"

#ifndef WIN32
#include <pthread.h>
#else
#define pthread_mutex_init(x)
#define pthread_mutex_destroy(x)
#define pthread_mutex_lock(x)
#define pthread_mutex_unlock(x)
#endif

class CSGObject;
class CIO;

// define reference counter macros
#define SG_REF(x) { if (x) (x)->ref(); }
#define SG_UNREF(x) { if (x) { if ((x)->unref()==0) (x)=0; } }

/** @brief Class SGObject is the base class of all shogun objects.
 *
 * Apart from dealing with reference counting that is used to manage shogung
 * objects in memory (erase unused object, avoid cleaning objects when they are
 * still in use), it provides interfaces for:
 *
 * -# parallel - to determine the number of used CPUs for a method (cf. CParallel)
 * -# io - to output messages and general i/o (cf. CIO)
 * -# version - to provide version information of the shogun version used (cf. CVersion)
 */
class CSGObject
{
public:
	inline CSGObject() : refcount(0)
	{
		set_global_objects();
		pthread_mutex_init(&ref_mutex, NULL);
	}

	inline CSGObject(const CSGObject& orig) : refcount(0), io(orig.io),
		parallel(orig.parallel), version(orig.version)
	{
		set_global_objects();
	}

    virtual ~CSGObject()
	{
		pthread_mutex_destroy(&ref_mutex);
		SG_UNREF(version);
		SG_UNREF(parallel);
		SG_UNREF(io);
	}

	/** increase reference counter
	 *
	 * @return reference count
	 */
	inline int32_t ref()
	{
		pthread_mutex_lock(&ref_mutex);
		++refcount;
		SG_DEBUG("ref() refcount %ld obj %s (%p) increased\n", refcount, this->get_name(), this);
		pthread_mutex_unlock(&ref_mutex);
		return refcount;
	}

	/** display reference counter
	 *
	 * @return reference count
	 */
	inline int32_t ref_count() const
	{
		SG_DEBUG("ref_count(): refcount %d, obj %s (%p)\n", refcount, this->get_name(), this);
		return refcount;
	}

	/** decrement reference counter and deallocate object if refcount is zero
	 * before or after decrementing it
	 *
	 * @return reference count
	 */
	inline int32_t unref()
	{
		pthread_mutex_lock(&ref_mutex);
		if (refcount==0 || --refcount==0)
		{
			SG_DEBUG("unref() refcount %ld, obj %s (%p) destroying\n", refcount, this->get_name(), this);
			pthread_mutex_unlock(&ref_mutex);
			delete this;
			return 0;
		}
		else
		{
			SG_DEBUG("unref() refcount %ld obj %s (%p) decreased\n", refcount, this->get_name(), this);
			pthread_mutex_unlock(&ref_mutex);
			return refcount;
		}
	}

	/** get the name of a kernel
	 *
	 * @return name of object
	 */
	virtual const char* get_name() const=0;

#ifdef HAVE_BOOST_SERIALIZATION
	/** Serialization Function: Convert object to a string
	 *
	 * @return string
	 */
	virtual std::string to_string() const;

	/** Serialization Function: Obtain object from string
	 *
	 * @param filename file name
	 */
	virtual void from_string(std::string str);

	/** Serialization Function: Save the object to file
	 *
	 * @param filename file name
	 */
	virtual void to_file(std::string filename) const;

	/** Serialization Function: Load the object from file
	 *
	 * @param filename file name
	 */
	virtual void from_file(std::string filename);

  protected:
	friend class boost::serialization::access;

	/** When the class Archive corresponds to an output archive, the & operator
	 * is defined similar to <<.  Likewise, when the class Archive is a type of
	 * input archive the & operator is defined similar to >>.
	 *
	 * @param ar output archive
	 * @param version_num version number
	 */
	template<class Archive>
		void serialize(Archive & ar, const unsigned int version_num)
		{
			//ar & test;
			SG_ERROR("SERIALIZING SGObject: nothing to do");
		}

#endif //HAVE_BOOST_SERIALIZATION


private:
	void set_global_objects();

private:
	int32_t refcount;
#ifndef WIN32
	pthread_mutex_t ref_mutex;
#endif

public:
	CIO* io;
	CParallel* parallel;
	CVersion* version;
};
#endif // __SGOBJECT_H__
