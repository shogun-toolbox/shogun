/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */

#include "base/SGObject.h"
#include "lib/io.h"
#include "base/Parallel.h"
#include "base/init.h"
#include "base/Version.h"
#include "lib/Parameter.h"

#include <stdlib.h>
#include <stdio.h>


#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
//BOOST_IS_ABSTRACT(CSGObject);
//BOOST_SERIALIZATION_ASSUME_ABSTRACT(shogun::CSGObject);
#endif //HAVE_BOOST_SERIALIZATION

namespace shogun
{
	class CMath;
	class CParallel;
	class CIO;
	class CVersion;

	extern CMath* sg_math;
	extern CParallel* sg_parallel;
	extern CIO* sg_io;
	extern CVersion* sg_version;

}

using namespace shogun;

void
CSGObject::set_global_objects(void)
{
	if (!sg_io || !sg_parallel || !sg_version)
	{
		fprintf(stderr, "call init_shogun() before using the library, dying.\n");
		exit(1);
	}

	SG_REF(sg_io);
	SG_REF(sg_parallel);
	SG_REF(sg_version);

	io=sg_io;
	parallel=sg_parallel;
	version=sg_version;

	m_parameters = new CParameter(io);
}

void
CSGObject::unset_global_objects(void)
{
	delete m_parameters;

	SG_UNREF(version);
	SG_UNREF(parallel);
	SG_UNREF(io);
}

void
CSGObject::params_print(const char* prefix)
{
	m_parameters->print(prefix);
}

bool
CSGObject::save(CFile* file, const char* prefix)
{
	return m_parameters->save(file, prefix);
}

bool
CSGObject::load(CFile* file, const char* prefix)
{
	return m_parameters->load(file, prefix);
}

void CSGObject::set_io(CIO* new_io)
{
	SG_UNREF(sg_io);
	sg_io=new_io;
	SG_REF(sg_io);
}

CIO* CSGObject::get_io()
{
	SG_REF(sg_io);
	return sg_io;
}

void CSGObject::set_parallel(CParallel* new_parallel)
{
	SG_UNREF(sg_parallel);
	sg_parallel=new_parallel;
	SG_REF(sg_parallel);
}

CParallel* CSGObject::get_parallel()
{
	SG_REF(sg_parallel);
	return sg_parallel;
}

void CSGObject::set_version(CVersion* new_version)
{
	SG_UNREF(sg_version);
	sg_version=new_version;
	SG_REF(sg_version);
}

CVersion* CSGObject::get_version()
{
	SG_REF(sg_version);
	return sg_version;
}


#ifdef HAVE_BOOST_SERIALIZATION
std::string CSGObject::to_string() const
{
	std::ostringstream s;
	boost::archive::text_oarchive oa(s);
	oa << this;
	return s.str();
}

void CSGObject::from_string(std::string str)
{
	std::istringstream is(str);
	boost::archive::text_iarchive ia(is);

	//cast away constness
	CSGObject* tmp = const_cast<CSGObject*>(this);

	ia >> tmp;
	*this = *tmp;
}

void CSGObject::to_file(std::string filename) const
{
	std::ofstream os(filename.c_str(), std::ios::binary);
	boost::archive::binary_oarchive oa(os);
	oa << this;
}

void CSGObject::from_file(std::string filename)
{
	std::ifstream is(filename.c_str(), std::ios::binary);
	boost::archive::binary_iarchive ia(is);
	CSGObject* tmp= const_cast<CSGObject*>(this);
	ia >> tmp;
}
#endif //HAVE_BOOST_SERIALIZATION
