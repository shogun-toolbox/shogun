#include "classifier/Classifier.h"

CClassifier::CClassifier() : labels(NULL)
{
}

CClassifier::~CClassifier()
{
}

REAL* CClassifier::test()
{
	if (labels)
	{
		INT num=labels->get_num_labels();
		assert(num>0);
		REAL* output=new REAL[num];

		assert(output);
		for (INT i=0; i<num; i++)
			output[i]=classify_example(i);

		return output;
	}

	return NULL;
}
