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

	struct svm_problem problem;
	struct svm_parameter param;
	struct svm_node* x_space;

	assert(get_labels() && get_labels()->get_num_labels());
	problem.l=get_labels()->get_num_labels();
	CIO::message("%d trainlabels\n", problem.l);

	problem.y=new double[problem.l];
	problem.x=new struct svm_node*[problem.l];
	x_space=new struct svm_node[2*problem.l];

	assert(problem.y);
	assert(problem.x);
	assert(x_space);

	for (int i=0; i<problem.l; i++)
	{
		problem.y[i]=get_labels()->get_label(i);
		problem.x[i]=&x_space[2*i];
		x_space[2*i].index=i;
		x_space[2*i+1].index=-1;
	}

	param.svm_type=C_SVC; // C SVM
	param.kernel_type = LINEAR;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-6;
	param.p = 0.1;
	param.shrinking = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.kernel=get_kernel();

	const char* error_msg = svm_check_parameter(&problem,&param);

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

	model = svm_train(&problem, &param);

	if (model)
	{
		assert(model->nr_class==2);
		assert(model->l>0);
		assert(model->SV);
		assert(model->nSV);
		assert(model->sv_coef && model->sv_coef[0]);

		int num_sv=model->l;

		create_new_model(num_sv);
		set_bias(-model->rho[0]);

		for (int i=0; i<num_sv; i++)
		{
			set_support_vector(i, (model->SV[i])->index);
			set_alpha(i, -model->sv_coef[0][i]);
		}

		delete[] problem.x;
		delete[] problem.y;
		delete[] x_space;

		return true;
	}
	else
		return false;
}
