/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Eleftherios Avramidis
 */

#include <shogun/modelselection/ModelSelectionParameter.h>

using namespace shogun;

CModelSelectionParameter::CModelSelectionParameter()
{
	init();
}


void CModelSelectionParameter::init()
{	
}

CModelSelectionParameter::~CModelSelectionParameter()
{
}

void CModelSelectionParameter::add_parameters(CModelSelectionParameter* params)
{
	ParametersMap parameters = params->filter(ParameterProperties::NONE);
	
	for (ParametersMap::iterator it=parameters.begin(); it!=parameters.end(); ++it)
	{
		double* temp_value = new double();
		*temp_value = params->get<double>(it->first.name());
		add_param<double>(it->first.name(), temp_value);
	}
}