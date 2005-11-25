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

	assert(get_labels());
	assert(get_features());
	INT num_train_labels=0;
	INT* train_labels=get_labels()->get_int_labels(num_train_labels);
	assert(train_labels);

	INT num_feat=features->get_num_features();
	INT num_vec=features->get_num_vectors();
	assert(num_vec==num_train_labels);

	INT* classidx=new INT[num_vec];

	INT i=0;
	INT num_neg=0;
	INT num_pos=num_feat-1;
	for (i=0; i<num_feat; i++)
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

	delete[] w;
	w=new REAL[num_feat];
	assert(w);

	REAL* mean_pos=new REAL[num_feat];
	assert(mean_pos);
	memset(mean_pos,0,num_feat*sizeof(REAL));

	REAL* mean_neg=new REAL[num_feat];
	assert(mean_neg);
	memset(mean_neg,0,num_feat*sizeof(REAL));

	REAL* scatter=new REAL[num_feat*num_feat];
	assert(scatter);
	memset(scatter,0,num_feat*num_feat*sizeof(REAL));


	//for (
	
//for ci= 1:2
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
//C.b = -0.5*sum(m.*C.w,1)' + log(prior);
//C.w = C.w(:,2) - C.w(:,1);
//C.b = C.b(2)-C.b(1);


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
