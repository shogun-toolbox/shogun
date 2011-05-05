#include <shogun/features/SimpleFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/DistantSegmentsKernel.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/io.h>
#include <stdio.h>

using namespace std;
using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

bool test_float_scalar()
{
	bool result=true;
	float64_t initialValue=1;
	float64_t anotherValue=2;

	Parameter* params=new Parameter();
	float64_t param1=initialValue;
	params->add(&param1, "param", "");

	float64_t param2=anotherValue;
	Parameter* params2=new Parameter();
	params2->add(&param2, "param", "");

	params->set_from_parameters(params2);

	result&=param1==anotherValue;

	delete params;
	delete params2;

	return result;
}

bool test_float_vector()
{
	bool result=true;
	float64_t initialValue=1;
	float64_t anotherValue=2;

	index_t max=3;
	Parameter* params=new Parameter();
	float64_t* param1=(float64_t*) malloc(sizeof(float64_t)*max);
	for (index_t i=0; i<max; ++i)
		param1[i]=initialValue;

	params->add_vector(&param1, &max, "param", "");

	float64_t* param2=(float64_t*) malloc(sizeof(float64_t)*max);
	for (index_t i=0; i<max; ++i)
		param2[i]=anotherValue;

	Parameter* params2=new Parameter();
	params2->add_vector(&param2, &max, "param", "");

	params->set_from_parameters(params2);

	for (index_t i=0; i<max; ++i)
		result&=param1[i]==anotherValue;

	free(param1);
	free(param2);
	delete params;
	delete params2;

	return result;
}

bool test_float_matrix()
{
	bool result=true;
	float64_t initialValue=1;
	float64_t anotherValue=2;

	index_t max=3;
	Parameter* params=new Parameter();
	float64_t* param1=new float64_t[max*max];
	for (index_t i=0; i<max; ++i)
		for (index_t j=0; j<max; ++j)
			param1[max*j+i]=initialValue;

	params->add_matrix(&param1, &max, &max, "param", "");

	float64_t* param2=new float64_t[max*max];
	for (index_t i=0; i<max; ++i)
		for (index_t j=0; j<max; ++j)
			param2[max*j+i]=anotherValue;

	Parameter* params2=new Parameter();
	params2->add_matrix(&param2, &max, &max, "param", "");

	params->set_from_parameters(params2);

	for (index_t i=0; i<max; ++i)
		for (index_t j=0; j<max; ++j)
			result&=param1[max*j+i]==anotherValue;

	delete param1;
	delete param2;
	delete params;
	delete params2;

	return result;
}

bool test_sgobject_scalar()
{
	bool result=true;

	Parameter* params=new Parameter();
	CSGObject* kernel=new CGaussianKernel(10, 10);
	SG_REF(kernel);
	params->add(&kernel, "kernel", "");

	CSGObject* kernel2=new CDistantSegmentsKernel(10, 10, 10);
	Parameter* params2=new Parameter();
	params2->add(&kernel2, "kernel", "");

	params->set_from_parameters(params2);

	result&=kernel==kernel2;

	delete params;
	delete params2;

	return result;
}

bool test_sgobject_vector()
{
	bool result=true;

	index_t max=3;
	Parameter* params=new Parameter();
	CSGObject** param1=new CSGObject*[max];
	for (index_t i=0; i<max; ++i)
	{
		param1[i]=new CDistantSegmentsKernel(1, 1, 1);
		SG_REF(param1[i]);
	}

	params->add_vector(&param1, &max, "param", "");

	CSGObject** param2=new CSGObject*[max];
	for (index_t i=0; i<max; ++i)
		param2[i]=new CDistantSegmentsKernel(2, 2, 2);

	Parameter* params2=new Parameter();
	params2->add_vector(&param2, &max, "param", "");

	params->set_from_parameters(params2);

	for (index_t i=0; i<max; ++i)
		result&=param1[i]==param2[i];

	delete param1;
	delete param2;
	delete params;
	delete params2;

	return result;
}

bool test_sgobject_matrix()
{
	bool result=true;

	index_t max=3;
	Parameter* params=new Parameter();
	CSGObject** param1=new CSGObject*[max*max];
	for (index_t i=0; i<max; ++i)
		for (index_t j=0; j<max; ++j)
		{
			param1[j*max+i]=new CDistantSegmentsKernel(1, 1, 1);
			SG_REF(param1[j*max+i]);
		}

	params->add_matrix(&param1, &max, &max, "param", "");

	CSGObject** param2=new CSGObject*[max*max];
	for (index_t i=0; i<max; ++i)
		for (index_t j=0; j<max; ++j)
			param2[j*max+i]=new CDistantSegmentsKernel(1, 1, 1);

	Parameter* params2=new Parameter();
	params2->add_matrix(&param2, &max, &max, "param", "");

	params->set_from_parameters(params2);

	for (index_t i=0; i<max; ++i)
		for (index_t j=0; j<max; ++j)
			result&=param1[j*max+i]==param2[j*max+i];

	delete param1;
	delete param2;
	delete params;
	delete params2;

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
		SG_SPRINT("SUCCESS!");
	else
		SG_SPRINT("FAILURE!");

	return 0;
}
