/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, Heiko Strathmann
 */

#include <shogun/lib/config.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/kernel/string/DistantSegmentsKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/PowerKernel.h>
#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/lib/SGStringList.h>

using namespace shogun;


void print_modsel_parameters(SGObject* object)
{
	SGStringList<char> modsel_params=object->get_modelsel_names();

	SG_SPRINT("Parameters of %s available for model selection:\n",
			object->get_name());

	char* type_string=SG_MALLOC(char, 100);
	for (index_t i=0; i<modsel_params.num_strings; ++i)
	{
		/* extract current name, ddescription and type, and print them */
		const char* name=modsel_params.strings[i].string;
		index_t index=object->get_modsel_param_index(name);
		TSGDataType type=object->m_model_selection_parameters->get_parameter(
				index)->m_datatype;
		type.to_string(type_string, 100);
		SG_SPRINT("\"%s\": \"%s\", %s\n", name,
				object->get_modsel_param_descr(name), type_string);
	}
	SG_FREE(type_string);

	SG_SPRINT("\n");
}

int main(int argc, char** argv)
{
#ifndef HAVE_LAPACK
	SGObject* object;

	object=new CLibSVM();
	print_modsel_parameters(object);

	object=new LibLinear();
	print_modsel_parameters(object);

	object=new CDistantSegmentsKernel();
	print_modsel_parameters(object);

	object=new GaussianKernel();
	print_modsel_parameters(object);

	object=new CPowerKernel();
	print_modsel_parameters(object);

	object=new CMinkowskiMetric();
	print_modsel_parameters(object);
#endif // HAVE_LAPACK

	return 0;
}

