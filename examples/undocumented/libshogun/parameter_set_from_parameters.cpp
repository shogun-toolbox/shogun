/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * DS-Kernel implementation Written (W) 2008 Sébastien Boisvert under GPLv3
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>
#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/kernel/GaussianKernel.h>

using namespace shogun;

int32_t max=3;
const float64_t initial_value=1;
const float64_t another_value=2;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

bool test_float_scalar()
{
	bool result=true;

	Parameter* original_parameter_list=new Parameter();
	float64_t original_parameter=initial_value;
	original_parameter_list->add(&original_parameter, "param", "");

	float64_t new_parameter=another_value;
	Parameter* new_parameter_list=new Parameter();
	new_parameter_list->add(&new_parameter, "param", "");

	original_parameter_list->set_from_parameters(new_parameter_list);

	result&=original_parameter==another_value;

	delete original_parameter_list;
	delete new_parameter_list;

	return result;
}

bool test_float_vector()
{
	bool result=true;

	Parameter* original_parameter_list=new Parameter();
	float64_t* original_parameter=SG_MALLOC(float64_t, max);
	SGVector<float64_t>::fill_vector(original_parameter, max, initial_value);

	original_parameter_list->add_vector(&original_parameter, &max, "param", "");

	float64_t* new_parameter=SG_MALLOC(float64_t, max);
	SGVector<float64_t>::fill_vector(new_parameter, max, another_value);

	Parameter* new_parameter_list=new Parameter();
	new_parameter_list->add_vector(&new_parameter, &max, "param", "");

	original_parameter_list->set_from_parameters(new_parameter_list);

	for (int32_t i=0; i<max; ++i)
		result&=original_parameter[i]==another_value;

	delete original_parameter;
	delete new_parameter;
	delete original_parameter_list;
	delete new_parameter_list;

	return result;
}

bool test_float_matrix()
{
	bool result=true;

	Parameter* original_parameter_list=new Parameter();
	float64_t* original_parameter=SG_MALLOC(float64_t, max*max);
	SGVector<float64_t>::fill_vector(original_parameter, max*max, initial_value);

	original_parameter_list->add_matrix(&original_parameter, &max, &max, "param", "");

	float64_t* new_parameter=SG_MALLOC(float64_t, max*max);
	SGVector<float64_t>::fill_vector(new_parameter, max*max, another_value);

	Parameter* new_parameter_list=new Parameter();
	new_parameter_list->add_matrix(&new_parameter, &max, &max, "param", "");

	original_parameter_list->set_from_parameters(new_parameter_list);

	for (int32_t i=0; i<max*max; ++i)
		result&=original_parameter[i]==another_value;

	delete original_parameter;
	delete new_parameter;
	delete original_parameter_list;
	delete new_parameter_list;

	return result;
}

bool test_sgobject_scalar()
{
	bool result=true;

	Parameter* original_parameter_list=new Parameter();
	CSGObject* original_parameter=new CGaussianKernel(10, 10);
	SG_REF(original_parameter);
	original_parameter_list->add(&original_parameter, "kernel", "");

	CSGObject* new_parameter=new CDistantSegmentsKernel(10, 10, 10);
	Parameter* new_parameter_list=new Parameter();
	new_parameter_list->add(&new_parameter, "kernel", "");

	/* note: old_parameter is SG_UNREF'ed, new one SG_REF'ed */
	original_parameter_list->set_from_parameters(new_parameter_list);

	result&=original_parameter==new_parameter;

	/* old original kernel was deleted by shogun's SG_UNREF */
	SG_UNREF(new_parameter);
	delete original_parameter_list;
	delete new_parameter_list;

	return result;
}

bool test_sgobject_vector()
{
	bool result=true;

	Parameter* original_parameter_list=new Parameter();
	CSGObject** original_parameter=SG_MALLOC(CSGObject*, max);
	for (int32_t i=0; i<max; ++i)
	{
		original_parameter[i]=new CDistantSegmentsKernel(1, 1, 1);
		SG_REF(original_parameter[i]);
	}

	original_parameter_list->add_vector(&original_parameter, &max, "param", "");

	CSGObject** new_parameter=SG_MALLOC(CSGObject*, max);
	for (int32_t i=0; i<max; ++i)
		new_parameter[i]=new CDistantSegmentsKernel(2, 2, 2);

	Parameter* new_parameter_list=new Parameter();
	new_parameter_list->add_vector(&new_parameter, &max, "param", "");

	/* note: old_parameters are SG_UNREF'ed, new ones SG_REF'ed */
	original_parameter_list->set_from_parameters(new_parameter_list);

	for (int32_t i=0; i<max; ++i)
		result&=original_parameter[i]==new_parameter[i];

	/* old original kernels were deleted by shogun's SG_UNREF */
	delete original_parameter;

	for (int32_t i=0; i<max; ++i)
		SG_UNREF(new_parameter[i]);

	delete new_parameter;
	delete original_parameter_list;
	delete new_parameter_list;

	return result;
}

bool test_sgobject_matrix()
{
	bool result=true;

	Parameter* original_parameter_list=new Parameter();
	CSGObject** original_parameter=SG_MALLOC(CSGObject*, max*max);
	for (int32_t i=0; i<max; ++i)
	{
		for (int32_t j=0; j<max; ++j)
		{
			original_parameter[j*max+i]=new CDistantSegmentsKernel(1, 1, 1);
			SG_REF(original_parameter[j*max+i]);
		}
	}

	original_parameter_list->add_matrix(&original_parameter, &max, &max, "param", "");

	CSGObject** new_parameter=SG_MALLOC(CSGObject*, max*max);
	for (int32_t i=0; i<max; ++i)
	{
		for (int32_t j=0; j<max; ++j)
			new_parameter[j*max+i]=new CDistantSegmentsKernel(1, 1, 1);
	}

	Parameter* new_parameter_list=new Parameter();
	new_parameter_list->add_matrix(&new_parameter, &max, &max, "param", "");

	/* note: old_parameters are SG_UNREF'ed, new ones SG_REF'ed */
	original_parameter_list->set_from_parameters(new_parameter_list);

	for (int32_t i=0; i<max; ++i)
	{
		for (int32_t j=0; j<max; ++j)
			result&=original_parameter[j*max+i]==new_parameter[j*max+i];
	}

	/* old original kernels were deleted by shogun's SG_UNREF */
	delete original_parameter;

	for (int32_t i=0; i<max*max; ++i)
		SG_UNREF(new_parameter[i]);

	delete new_parameter;
	delete original_parameter_list;
	delete new_parameter_list;

	return result;
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	bool result=true;

	/* test wheater set_from_parameters works for these types */
	result&=test_float_scalar();
	result&=test_sgobject_scalar();
	result&=test_sgobject_vector();
	result&=test_sgobject_matrix();
	result&=test_float_matrix();
	result&=test_float_vector();

	if (result)
		SG_SPRINT("SUCCESS!\n")
	else
		SG_SPRINT("FAILURE!\n")

	exit_shogun();

	return 0;
}
