#include "classifier/svm/LibSVM.h"
#include "lib/io.h"
#include <ctype.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
void svm_read_problem(const char *filename, svm_problem& prob, svm_parameter& param, svm_node* x_space)
{
	int elements, max_index, i, j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				++prob.l;
				// fall through,
				// count the '-1' element
			case ':':
				++elements;
				break;
			case EOF:
				goto out;
			default:
				;
		}
	}
out:
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		double label;
		prob.x[i] = &x_space[j];
		fscanf(fp,"%lf",&label);
		prob.y[i] = label;
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
			++j;
		}	
out2:
		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0)
		param.gamma = 1.0/max_index;

	fclose(fp);
}

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

	svm_read_problem("/tmp/libsvmtraindataset", problem, param, x_space);
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
			set_alpha(i, model->sv_coef[0][i]);
		}

		delete[] problem.x;
		delete[] problem.y;
		delete[] x_space;

		return true;
	}
	else
		return false;
}
