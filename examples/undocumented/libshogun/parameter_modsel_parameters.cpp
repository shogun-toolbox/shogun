/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/lib/config.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/lib/SGStringList.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void print_modsel_parameters(CSGObject* object)
{
	SG_SPRINT("Parameters of %s available for model selection:\n",
			object->get_name());

	char* type_string=SG_MALLOC(char, 100);
	for (index_t i=0; i<modsel_params.num_strings; ++i)
	{
		/* extract current name, ddescription and type, and print them */
		const char* name=modsel_params.strings[i].string;
		index_t index=object->get_modsel_param_index(name);
		type.to_string(type_string, 100);
		SG_SPRINT("\"%s\": \"%s\", %s\n", name,
				object->get_modsel_param_descr(name), type_string);
	}
	SG_FREE(type_string);

	SG_SPRINT("\n");
}

int main(int argc, char** argv)
{
	init_shogun(&print_message);

#ifndef HAVE_LAPACK
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
#endif // HAVE_LAPACK

	exit_shogun();
	return 0;
}

