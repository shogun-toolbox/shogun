/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/base/Parameter.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/base/ParameterMap.h>
#include <unistd.h>

using namespace shogun;

class CTestClassOld : public CSGObject
{
public:
	CTestClassOld()
	{

	}

	virtual const char* get_name() const { return "TestClassOld"; }
};

class CTestClassNew : public CSGObject
{
public:
	CTestClassNew()
	{
		m_number=3;
		m_parameters->add(&m_number, "m_number", "");

		/* this parameter is new in this version, mapping from "nowhere" */
		m_parameter_map->put(
				new SGParamInfo("m_number", CT_SCALAR, ST_NONE, PT_INT32, 1),
				new SGParamInfo()
		);

		/* note that dropped parameters need not be considered, just ignored */

		/* needed if more than one element */
		m_parameter_map->finalize_map();
	}

	int32_t m_number;

	virtual const char* get_name() const { return "TestClassNew"; }

};

void test()
{
	char filename_tmp[] = "migration_buggy_test.XXXXXX";
    int fd = mkstemp(filename_tmp);
    ASSERT(fd != -1);
	char* filename = filename_tmp;

	/* create one instance of each class */
	CTestClassOld* old_instance=new CTestClassOld();
	CTestClassNew* new_instance=new CTestClassNew();

	CSerializableAsciiFile* file;

	/* serialize int instance, use custom parameter version */
	file=new CSerializableAsciiFile(filename, 'w');
	old_instance->save_serializable(file, "", 0);
	file->close();
	SG_UNREF(file);

	/* de-serialize float instance, use custom parameter version 2 which means
	 * that the class is already one step further and the parameter which was
	 * added in version 1 has to be migrated (which is done automatically) */
	// change this version to two once it works!
	file=new CSerializableAsciiFile(filename, 'r');
	new_instance->load_serializable(file, "", 1);
	file->close();
	SG_UNREF(file);

	SG_UNREF(old_instance);
	SG_UNREF(new_instance);
	SG_UNREF(file);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

