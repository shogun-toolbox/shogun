#include "classifier/Perceptron.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CPerceptron::CPerceptron() : CLinearClassifier(), learn_rate(0.1), max_iter(10000000)
{
}


CPerceptron::~CPerceptron()
{
}

bool CPerceptron::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());
	bool converged=false;
	INT iter=0;
	INT num_train_labels=0;
	INT* train_labels=get_labels()->get_int_labels(num_train_labels);
	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();

	ASSERT(num_vec==num_train_labels);
	delete[] w;
	w=new REAL[num_feat];
	ASSERT(w);
	REAL* output=new REAL[num_vec];
	ASSERT(output);

	//start with uniform w, bias=0
	bias=0;
	for (INT i=0; i<num_feat; i++)
		w[i]=1.0/num_feat;

	//loop till we either get everything classified right or reach max_iter
	while (!converged && iter<max_iter)
	{
		converged=true;
		for (INT i=0; i<num_vec; i++)
			output[i]=classify_example(i);

		for (INT i=0; i<num_vec; i++)
		{
			if (CMath::sign<REAL>(output[i]) != train_labels[i])
			{
				converged=false;
				INT vlen;
				bool vfree;
				double* vec=features->get_feature_vector(i, vlen, vfree);

				bias+=learn_rate*train_labels[i];
				for (INT j=0; j<num_feat; j++)
					w[j]+=  learn_rate*train_labels[i]*vec[j];

				features->free_feature_vector(vec, i, vfree);
			}
		}

		iter++;
	}
	delete[] output;
	delete[] train_labels;

	return false;
}
