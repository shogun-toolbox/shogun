/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/kernel/DistantSegmentsKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_modsel_parameters(CSGObject* object)
{
	SGVector<char*> modsel_params=object->get_modelsel_names();

	SG_SPRINT("Parameters of %s available for model selection:\n",
			object->get_name());

	char* type_string=SG_MALLOC(char, 100);
	for (index_t i=0; i<modsel_params.vlen; ++i)
	{
		/* extract current name, ddescription and type, and print them */
		const char* name=modsel_params.vector[i];
		index_t index=object->get_modsel_param_index(name);
		TSGDataType type=object->m_model_selection_parameters->get_parameter(
				index)->m_datatype;
		type.to_string(type_string, 100);
		SG_SPRINT("\"%s\": \"%s\", %s\n", name,
				object->get_modsel_param_descr(name), type_string);
	}
	SG_FREE(type_string);

	modsel_params.destroy_vector();
	SG_SPRINT("\n");
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);

	CSGObject* object;

	object=new CLibSVM();
	print_modsel_parameters(object);
	SG_UNREF(object);

	object=new CLibLinear();
	print_modsel_parameters(object);
	SG_UNREF(object);

	object=new CDistantSegmentsKernel();
	print_modsel_parameters(object);
	SG_UNREF(object);

	object=new CGaussianKernel();
	print_modsel_parameters(object);
	SG_UNREF(object);

	object=new CPowerKernel();
	print_modsel_parameters(object);
	SG_UNREF(object);

	object=new CMinkowskiMetric();
	print_modsel_parameters(object);
	SG_UNREF(object);

	exit_shogun();
	return 0;
}

