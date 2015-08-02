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
#include <shogun/lib/SGVector.h>
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
		m_number_1=1;
		m_number_2=2;
		m_parameters->add(&m_number_1, "m_number_1", "");
		m_parameters->add(&m_number_2, "m_number_2", "");
	}

	int32_t m_number_1;
	int32_t m_number_2;

	virtual const char* get_name() const { return "TestClassOld"; }
};

class CTestClassNew : public CSGObject
{
public:
	CTestClassNew()
	{
		m_number=0;
		m_parameters->add(&m_number, "m_number", "");

		/* number_1 in old version will become new, merged number */
		m_parameter_map->put(
				new SGParamInfo("m_number", CT_SCALAR, ST_NONE, PT_INT32, 1),
				new SGParamInfo("m_number_1", CT_SCALAR, ST_NONE, PT_INT32, 0)
		);

		/* Note that here, two mappings for one parameter are added. This means
		 * that m_number both depends on m_number_1 and m_number_2 */
		m_parameter_map->put(
				new SGParamInfo("m_number", CT_SCALAR, ST_NONE, PT_INT32, 1),
				new SGParamInfo("m_number_2", CT_SCALAR, ST_NONE, PT_INT32, 0)
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

		if (*target==SGParamInfo("m_number", CT_SCALAR, ST_NONE, PT_INT32, 1))
		{
			/* one to one migration may not be used here because two parameters
			 * are merged into one parameter. Here the new parameter will
			 * contain the sum of the two old ones. */

			/* generate type of target structure */
			TSGDataType type(target->m_ctype, target->m_stype, target->m_ptype);

			/* find elements that are needed for migration, in this case the
			 * two numbers of the base */
			char* name_1=(char*) "m_number_1";
			char* name_2=(char*) "m_number_2";

			/* dummy elements for searching */
			TParameter* t_1=new TParameter(&type, NULL, name_1, "");
			TParameter* t_2=new TParameter(&type, NULL, name_2, "");
			index_t i_1=CMath::binary_search(param_base->get_array(),
					param_base->get_num_elements(), t_1);
			index_t i_2=CMath::binary_search(param_base->get_array(),
					param_base->get_num_elements(), t_2);

			delete t_1;
			delete t_2;

			/* gather search results and tell them that they are to be deleted
			 * because they will be replaced */
			ASSERT(i_1>=0 && i_2>=0);
			TParameter* to_migrate_1=param_base->get_element(i_1);
			TParameter* to_migrate_2=param_base->get_element(i_2);
			to_migrate_1->m_delete_data=true;
			to_migrate_2->m_delete_data=true;

			/* create result structure and allocate data for it */
			result=new TParameter(&type, NULL, target->m_name,
					"New description");

			/* scalar value has length one */
			result->allocate_data_from_scratch(1, 1);

			/* merged element contains sum of both to be merged elements */
			*((int32_t*)result->m_parameter)=
					*((int32_t*)to_migrate_1->m_parameter)+
					*((int32_t*)to_migrate_2->m_parameter);
		}

		if (result)
			return result;
		else
			return CSGObject::migrate(param_base, target);
	}
};

void test_migration()
{
	char filename_tmp[] = "migration_multiple_dep_test.XXXXXX";
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

	/* de-serialize float instance, use custom parameter version */
	file=new CSerializableAsciiFile(filename, 'r');
	new_instance->load_serializable(file, "", 1);
	file->close();
	SG_UNREF(file);

	/* check that merged number is sum old to be merged ones */
	SG_SPRINT("checking \"m_number\":\n");
	SG_SPRINT("\t%d==%d+%d\n", new_instance->m_number, old_instance->m_number_1,
			old_instance->m_number_2);
	ASSERT(new_instance->m_number==old_instance->m_number_1+
			old_instance->m_number_2);

	SG_UNREF(old_instance);
	SG_UNREF(new_instance);
	SG_UNREF(file);
	unlink(filename);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	/* this is a more complex example, where a parameter is based on two
	 * old parameter */
	test_migration();

	exit_shogun();

	return 0;
}

