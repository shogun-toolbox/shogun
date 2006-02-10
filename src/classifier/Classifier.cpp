#include "classifier/Classifier.h"

CClassifier::CClassifier() : labels(NULL)
{
}

CClassifier::~CClassifier()
{
}

CLabels* CClassifier::classify(CLabels* output)
{
	if (labels)
	{
		INT num=labels->get_num_labels();
		ASSERT(num>0);

		if (!output)
			output=new CLabels(num);

		ASSERT(output);
		for (INT i=0; i<num; i++)
			output->set_label(i, classify_example(i));

		return output;
	}

	return NULL;
}
