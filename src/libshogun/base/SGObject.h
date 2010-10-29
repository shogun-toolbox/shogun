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

#include "lib/io.h"
#include "lib/SGSerializable.h"
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

/** \namespace shogun
 * @brief all of classes and functions are contained in the shogun namespace
 */
namespace shogun
{
class IO;
class Parallel;
class Version;

/** @brief Class SGObject is the base class of all shogun objects.
 *
 * Apart from dealing with reference counting that is used to manage shogung
 * objects in memory (erase unused object, avoid cleaning objects when they are
 * still in use), it provides interfaces for:
 *
 * -# parallel - to determine the number of used CPUs for a method (cf. Parallel)
 * -# io - to output messages and general i/o (cf. IO)
 * -# version - to provide version information of the shogun version used (cf. Version)
 */
class CSGObject :public CSGSerializable
{
public:
	inline CSGObject()
	{
		set_global_objects();
		pthread_mutex_init(&ref_mutex, NULL);
	}

	inline CSGObject(const CSGObject& orig)
		:io(orig.io), parallel(orig.parallel), version(orig.version)
	{
		set_global_objects();
	}

    virtual ~CSGObject()
	{
		pthread_mutex_destroy(&ref_mutex);
		unset_global_objects();
	}

	/** Returns the name of the SGSerializable instance.  It MUST BE
	 *  the CLASS NAME without the prefixed `C'.
	 *
	 *  @return name of the SGSerializable
	 */
	virtual const char* get_name(void) const = 0;

	/** set the io object
	 *
	 * @param io io object to use
	 */
	void set_io(IO* io);

	/** get the io object
	 *
	 * @return io object
	 */
	IO* get_io();

	/** set the parallel object
	 *
	 * @param parallel parallel object to use
	 */
	void set_parallel(Parallel* parallel);

	/** get the parallel object
	 *
	 * @return parallel object
	 */
	Parallel* get_parallel();

	/** set the version object
	 *
	 * @param version version object to use
	 */
	void set_version(Version* version);

	/** get the version object
	 *
	 * @return version object
	 */
	Version* get_version();

private:
	void set_global_objects(void);
	void unset_global_objects(void);

#ifndef WIN32
	pthread_mutex_t ref_mutex;
#endif

public:
	IO* io;
	Parallel* parallel;
	Version* version;
};
}
#endif // __SGOBJECT_H__
