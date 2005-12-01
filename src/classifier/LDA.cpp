#ifdef HAVE_LAPACK

#include "classifier/LDA.h"
#include "features/Labels.h"
#include "lib/Mathmatics.h"

CLDA::CLDA() : CLinearClassifier(), learn_rate(0.1), max_iter(10000000)
{
}


CLDA::~CLDA()
{
}

bool CLDA::train()
{
	REAL prior=1.0;
	REAL gamma=0;

	assert(get_labels());
	assert(get_features());
	INT num_train_labels=0;
	INT* train_labels=get_labels()->get_int_labels(num_train_labels);
	assert(train_labels);

	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();
	assert(num_vec==num_train_labels);

	INT* classidx=new INT[num_vec];
	assert(classidx);

	INT i=0;
	INT j=0;
	INT num_neg=0;
	INT num_pos=num_feat-1;
	for (i=0; i<num_train_labels; i++)
	{
		if (train_labels[i]==-1)
			classidx[num_neg++]=i;
		else if (train_labels[i]==+1)
			classidx[num_pos++]=i;
		else
		{
			CIO::message(M_ERROR, "found label != +/- 1 bailing...");
			return false;
		}
	}

	if (num_neg<=0 && num_pos<=0)
	{
		CIO::message(M_ERROR, "whooooo ? only a single class found\n");
		return false;
	}

	INT* classidx_neg=classidx;
	INT* classidx_pos=&classidx[num_neg];

	delete[] w;
	w=new REAL[num_feat];
	assert(w);

	REAL* mean_neg=new REAL[num_feat];
	assert(mean_neg);
	memset(mean_neg,0,num_feat*sizeof(REAL));

	REAL* mean_pos=new REAL[num_feat];
	assert(mean_pos);
	memset(mean_pos,0,num_feat*sizeof(REAL));

	REAL* scatter=new REAL[num_feat*num_feat];
	assert(scatter);
	memset(scatter,0,num_feat*num_feat*sizeof(REAL));

	REAL* buffer=new REAL[num_feat*CMath::max(num_neg, num_pos)];
	assert(buffer);

	//neg
	for (i=0; i<num_neg; i++)
	{
		INT vlen;
		bool vfree;
		double* vec=features->get_feature_vector(classidx_neg[i], vlen, vfree);

		for (j=0; j<vlen; j++)
		{
			mean_neg[j]+=vec[j];
			buffer[num_feat*i+j]=vec[j];
		}

		features->free_feature_vector(vec, classidx_neg[i], vfree);
	}

	for (j=0; j<num_feat; j++)
		mean_neg[j]/=num_neg;

	for (i=0; i<num_neg; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_neg[j];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_feat, num_feat, num_feat, 1.0, buffer, num_feat, buffer, num_feat, 1.0, scatter, num_feat);
	
	//mean pos
	for (i=0; i<num_pos; i++)
	{
		INT vlen;
		bool vfree;
		double* vec=features->get_feature_vector(classidx_pos[i], vlen, vfree);

		for (j=0; j<vlen; j++)
		{
			mean_neg[j]+=vec[j];
			buffer[num_feat*i+j]=vec[j];
		}

		features->free_feature_vector(vec, classidx_pos[i], vfree);
	}

	for (j=0; j<num_feat; j++)
		mean_neg[j]/=num_neg;

	for (i=num_neg; i<num_train_labels; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_neg[j];
	}
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_feat, num_feat, num_feat, 1.0, buffer, num_feat, buffer, num_feat, 1.0, scatter, num_feat);

	REAL trace=CMath::trace(scatter, num_feat, num_feat)/num_train_labels;

	for (i=0; i<num_feat*num_feat; i++)
		scatter[i]=(1-gamma)*scatter[i]/num_train_labels;

	for (i=0; i<num_feat; i++)
		scatter[i*num_feat+i]+= trace*gamma/num_feat;
	
	REAL* p= CMath::pinv(scatter, num_feat, num_feat, NULL);
	bias=log(prior);
	memcpy(buffer,mean_neg,sizeof(REAL)*num_feat);
	cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit, num_feat, scatter, num_feat, mean_neg, 1);
	bias-=0.5*CMath::dot(mean_neg, buffer, num_feat);
	cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasUnit, num_feat, scatter, num_feat, mean_pos, 1);
	bias+=0.5*CMath::dot(mean_pos, buffer, num_feat);

	for (i=0; i<num_feat; i++)
		w[i]=mean_pos[i]-mean_neg[i];

	delete[] train_labels;
	return false;
}

//priorP = ones(nClasses,1)/nClasses;
//
//d= size(xTr,1);
//m= zeros(d, nClasses);
//Sq= zeros(d, d);
//for ci= 1:nClasses,
//  cli= clInd{ci};
//  m(:,ci)= mean(xTr(:,cli),2);
//  yc= xTr(:,cli) - m(:,ci)*ones(1,N(ci));
//  Sq= Sq + yc*yc';
//end
//Sq= Sq/(sum(N)-1);
//Sq = (1-gamma)*Sq + gamma/d*trace(Sq)*eye(d);
//Sq = pinv(Sq);
//
//C.w = Sq*m;
//C.b = -0.5*sum(m.*C.w,1)' + log(priorP);
//
//if nClasses==2
//  C.w = C.w(:,2) - C.w(:,1);
//  C.b = C.b(2)-C.b(1);
//end
#endif
