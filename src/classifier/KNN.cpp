#include "classifier/KNN.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CKNN::CKNN(): k(3), num_classes(0), num_train_labels(0), train_labels(NULL)
{
}


CKNN::~CKNN()
{
	delete[] train_labels;
}

bool CKNN::train()
{
	assert(CKernelMachine::get_labels());
	train_labels=CKernelMachine::get_labels()->get_int_labels(num_train_labels);

	assert(train_labels);
	assert(num_train_labels>0);

	int max_class=train_labels[0];
	int min_class=train_labels[0];

	int i;
	for (i=1; i<num_train_labels; i++)
	{
		max_class=math.max(max_class, train_labels[i]);
		min_class=math.min(min_class, train_labels[i]);
	}

	for (i=0; i<num_train_labels; i++)
		train_labels[i]-=min_class;

	min_label=min_class;
	num_classes=max_class-min_class;

	CIO::message("num_classes: %d num_train: %d\n", num_classes, num_train_labels);
	return true;
}

REAL* CKNN::test()
{
	assert(CKernelMachine::get_kernel());
	assert(CKernelMachine::get_labels());
	assert(CKernelMachine::get_labels()->get_num_labels());

	int num_lab=CKernelMachine::get_labels()->get_num_labels();
	CKernel* kernel=CKernelMachine::get_kernel();

	REAL* dists=new REAL[num_train_labels];
	INT* train_lab=new INT[num_train_labels];
	REAL* output=new REAL[num_lab];
	INT* classes=new INT[num_classes];

	assert(dists);
	assert(train_lab);
	assert(output);
	assert(classes);

	CIO::message("%d test examples\n", num_lab);
	for (int i=0; i<num_lab; i++)
	{
		if ( (i% (num_lab/10+1))== 0)
			CIO::message("%i%%..",100*i/(num_lab+1));

		int j;
		for (j=0; j<num_train_labels; j++)
		{
			//copy back train labels and compute distance
			train_lab[j]=train_labels[j];
			dists[j]=kernel->kernel(j,i);
		}

		//sort the distance vector for test example j to all train examples
		//classes[1..k] then holds the classes for minimum distance
		math.qsort(dists, classes, num_train_labels);

		//compute histogram of class outputs
		for (j=0; j<num_classes; j++)
			classes[j]=0;

		for (j=0; j<k; j++)
			classes[train_lab[j]]++;


		//choose the class that got 'outputted' most often
		INT out_idx=0;
		INT out_max=0;

		for (j=0; j<num_classes; j++)
		{
			if (out_max< classes[j])
			{
				out_idx= j;
				out_max= classes[j];
			}
		}

		output[i]=out_idx;
	}

	delete[] dists;
	delete[] train_lab;
	delete[] classes;

	return output;
}

bool CKNN::load(FILE* srcfile)
{
	return false;
}

bool CKNN::save(FILE* dstfile)
{
	return false;
}
