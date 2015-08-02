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

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

class CTestClassOld : public CSGObject
{
public:
	CTestClassOld()
	{
		m_number_to_drop=10;
		m_parameters->add(&m_number_to_drop, "m_number_to_drop", "");

		m_number_to_keep=10;
		m_parameters->add(&m_number_to_keep, "m_number_to_keep", "");
	}

	int32_t m_number_to_drop;
	int32_t m_number_to_keep;

	virtual const char* get_name() const { return "TestClassOld"; }
};

class CTestClassNew : public CSGObject
{
public:
	CTestClassNew()
	{
		m_number=3;
		m_parameters->add(&m_number, "m_number", "");

		m_number_new=4;
		m_parameters->add(&m_number_new, "m_number_new", "");

		/* change name of to be kept number */
		m_parameter_map->put(
				new SGParamInfo("m_number", CT_SCALAR, ST_NONE, PT_INT32, 1),
				new SGParamInfo("m_number_to_keep", CT_SCALAR, ST_NONE, PT_INT32, 0)
		);

		/* this parameter is new in this version, mapping from "nowhere" */
		m_parameter_map->put(
				new SGParamInfo("m_number_new", CT_SCALAR, ST_NONE, PT_INT32, 1),
				new SGParamInfo()
		);

		/* note that dropped parameters need not be considered, just ignored */

		/* needed if more than one element */
		m_parameter_map->finalize_map();
	}

	int32_t m_number;
	int32_t m_number_new;

	virtual const char* get_name() const { return "TestClassNew"; }

	virtual TParameter* migrate(DynArray<TParameter*>* param_base,
				const SGParamInfo* target)
	{
		TParameter* result=NULL;
		TParameter* to_migrate=NULL;

		if (*target==SGParamInfo("m_number", CT_SCALAR, ST_NONE, PT_INT32, 1))
		{
			/* specify name change here (again, was also done in mappings) */
			char* old_name=(char*) "m_number_to_keep";
			one_to_one_migration_prepare(param_base, target, result,
					to_migrate, old_name);

			/* here: simply copy data because nothing has changed */
			*((int32_t*)result->m_parameter)=
					*((int32_t*)to_migrate->m_parameter);
		}

		/* note there has to be no case distinction for the new parameter */

		if (result)
			return result;
		else
			return CSGObject::migrate(param_base, target);
	}
};

void check_equalness(CTestClassOld* old_instance,
		CTestClassNew* new_instance)
{
	/* number */
	SG_SPRINT("checking \"m_number\":\n");
	SG_SPRINT("\t%d==%d\n", old_instance->m_number_to_keep,
			new_instance->m_number);
	ASSERT(old_instance->m_number_to_keep==new_instance->m_number);

	/* new element */
	SG_SPRINT("checking \"m_number_new\":\n");
	SG_SPRINT("\t%d\n", new_instance->m_number_new);
}

void test_migration()
{
	char filename_template[] = "migration_dropping_test.XXXXXX";
        int fd = mkstemp(filename_template);
        ASSERT(fd != -1);
	char* filename = filename_template;

	/* create one instance of each class */
	CTestClassOld* old_instance=new CTestClassOld();
	CTestClassNew* new_instance=new CTestClassNew();

	CSerializableAsciiFile* file;

	/* serialize int instance, use custom parameter version */
	file=new CSerializableAsciiFile(filename, 'w');
	old_instance->save_serializable(file, "", 0);
	file->close();
	SG_UNREF(file);

	/* de-serialize float instance, use custom parameter version */
	file=new CSerializableAsciiFile(filename, 'r');
	new_instance->load_serializable(file, "", 1);
	file->close();
	SG_UNREF(file);

	/* assert that content is equal */
	check_equalness(old_instance, new_instance);

	SG_UNREF(old_instance);
	SG_UNREF(new_instance);
	SG_UNREF(file);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_migration();

	exit_shogun();

	return 0;
}

