#include "classifier/svm/LibSVM.h"
#include "lib/io.h"

CLibSVM::CLibSVM()
{
	model=NULL;
}

CLibSVM::~CLibSVM()
{
	free(model);
}

bool CLibSVM::train()
{
	free(model);

	svm_problem problem;
	svm_parameter param;

	assert(get_labels() && get_labels()->get_num_labels());
	problem.l=get_labels()->get_num_labels();
	problem.y=new double[problem.l];
	problem.x=new svm_node*[problem.l];

	for (int i=0; i<problem.l; i++)
	{
		problem.y[i]=get_labels()->get_label(i);
		problem.x[i]=new svm_node[0];
		problem.x[i]->index=i;
		CIO::message("%f\n",problem.y[i]);
	}

	param.svm_type=C_SVC; // C SVM
	param.C=C;
	param.shrinking=0;
	param.cache_size=100; //for testing 100M cache (grrghh)
	param.eps=1e-6;

	model = svm_train(&problem, &param);

	if (model)
	{
		CIO::message("nr_class:%d\n", model->nr_class);
		assert(model->nr_class==2);
		assert(model->nSV && (*(model->nSV))==model->l);

		create_new_model(model->l);
		
		for (int i=0; i<model->l; i++)
		{
			set_support_vector(i, model->SV[i]->index);
			set_alpha(i, (*(model->sv_coef))[i]);
			set_bias(*(model->rho));
		}
		return true;
	}
	else
		return false;
}
