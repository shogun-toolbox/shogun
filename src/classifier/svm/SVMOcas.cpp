/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Vojtech Franc 
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/Time.h"
#include "base/Parallel.h"
#include "classifier/SparseLinearClassifier.h"
#include "classifier/svm/SVMOcas.h"
#include "classifier/svm/qpssvmlib.h"
#include "features/SparseFeatures.h"
#include "features/Labels.h"

const INT qpsolver_maxit=10000000;
DREAL* thread_times;

#define INDEX2(ROW,COL,NUM_ROWS) ((COL)*(NUM_ROWS)+(ROW))
#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define ABS(A) ((A) < 0 ? -(A) : (A))

DREAL *H;
INT BufSize;

struct thread_params_output
{
	DREAL* thread_time;
	DREAL C;
	DREAL* proj;
	DREAL* proj_old;
	DREAL* Ci;
	DREAL* Bi;
	DREAL* W;
	CSparseFeatures<DREAL>* data_X;
	INT start;
	INT end;
};

struct thread_params_add
{
	DREAL t1;
	DREAL t2;
	DREAL new_b;
	DREAL xi;
	DREAL err;
	DREAL* new_a;
	DREAL* proj;
	DREAL* proj_old;
	CSparseFeatures<DREAL>* data_X;
	INT start;
	INT end;
};

static void* compute_output_helper(void* p)
{
	struct thread_params_output* params = (struct thread_params_output*) p;
	DREAL* thread_time=params->thread_time;
	DREAL C=params->C;
	DREAL* proj=params->proj;
	DREAL* proj_old=params->proj_old;
	DREAL* Ci=params->Ci;
	DREAL* Bi=params->Bi;
	DREAL* W=params->W;
	CSparseFeatures<DREAL>* data_X=params->data_X;
	INT start=params->start;
	INT end=params->end;
	INT i;

	DREAL time=CTime::get_curtime();
	for (i=start; i<end; i++)
	{
		proj_old[i] = proj[i];
		proj[i] =  data_X->dense_dot(1.0, i, W, data_X->get_num_features(), 0.0);

		Ci[i] = C*(1-proj_old[i]);
		Bi[i] = C*(proj_old[i] - proj[i]);
	}

	(*thread_time)+=CTime::get_curtime()-time;

	return NULL;
}

static void* add_helper(void* p)
{
	struct thread_params_add* params = (struct thread_params_add*) p;
	DREAL t1=params->t1;
	DREAL t2=params->t2;
	DREAL* proj=params->proj;
	DREAL* proj_old=params->proj_old;
	DREAL* new_a = params->new_a;
	CSparseFeatures<DREAL>* data_X=params->data_X;
	INT start=params->start;
	INT end=params->end;

	for(INT i=start; i < end; i++ ) {
		DREAL wx = 1 - (proj_old[i]*(1-t2) + t2*proj[i]);
		proj[i] = proj_old[i]*(1-t1) + t1*proj[i];

		if( 1-proj[i]>=0 ) 
		{
			params->xi += 1-proj[i];
			params->err++;
		}

		if( wx >=0 ) {
			params->new_b++;
			data_X->add_to_dense_vec(1.0, i, new_a, data_X->get_num_features());  /* new_a = new_a + data_X(:,i)  */
		}
	}

	return NULL;
}


CSVMOcas::CSVMOcas(E_SVM_TYPE type) : CSparseLinearClassifier(), C1(1), C2(1),
	epsilon(1e-5), method(type)
{
}

CSVMOcas::CSVMOcas(DREAL C, CSparseFeatures<DREAL>* traindat, CLabels* trainlab) 
: CSparseLinearClassifier(), C1(C), C2(C), epsilon(1e-5)
{
	method=E_SVMOCAS;
	CSparseLinearClassifier::features=traindat;
	CClassifier::labels=trainlab;
}


CSVMOcas::~CSVMOcas()
{
}

bool CSVMOcas::train()
{
	ASSERT(get_labels());
	ASSERT(get_features());

	//INT num_train_labels=get_labels()->get_num_labels();
	//INT num_feat=features->get_num_features();
	//INT num_vec=features->get_num_vectors();

	//ASSERT(num_vec==num_train_labels);

	//delete[] w;
	//w=new DREAL[num_feat];
	//ASSERT(w);
	//bias=0;

	DREAL C, TolRel, TolAbs, GradVal, Bsum, t, t1, t2, QPBound;
	DREAL *W, *b, *alpha, *new_a, *diag_H, **A_val, *Wold, *proj, *proj_old;
	DREAL *Ci, *Bi;
	DREAL Q_D, Q_P, xi, err, wx, new_b, norm_W2, aa, norm_a2, QPSolverTolRel, dummy, A0, B0;
	INT MaxIter, nDim, nData, nIter, nSel, *A_len, **A_idx;
	INT i, j, k, ptr, nItems;
	WORD *I;
	INT exitFlag;
	INT solver_flag;
	CSparseFeatures<DREAL> *data_X;
	DREAL *data_y;
	DREAL *hpf;
	INT *hpi;

	/* parallelization via threads */
	pthread_t* threads;
	thread_params_output* params_output;
	thread_params_add* params_add;
	INT* thread_slices;

	/* timing variables */
	DREAL time_diff;
	DREAL wait_time;
	DREAL solver_time;
	DREAL add_time;
	DREAL sort_time;
	DREAL output_time;
	DREAL init_time;
	DREAL total_time;
	INT num_threads = parallel.get_num_threads();

	/* get input arguments */
	//data_X = (mxArray*)prhs[0];
	//data_y = (DREAL*) mxGetPr(prhs[1]);
	//C = (DREAL)mxGetScalar(prhs[2]);
	//TolRel = (DREAL)mxGetScalar(prhs[4]);
	//TolAbs = (DREAL)mxGetScalar(prhs[5]);
	//QPBound = (DREAL)mxGetScalar(prhs[6]);
	//MaxIter = mxIsInf( mxGetScalar(prhs[7])) ? INT_MAX : (uINT32_T)mxGetScalar(prhs[7]);
	//BufSize = (uINT32_T)mxGetScalar(prhs[8]);
	//num_threads = (uINT32_T)mxGetScalar(prhs[9]);

	/* Precision of the inner QP solver.
	ToDo: Better strategy to set the precision adaptively, e.g. starting from
	high (imprecise) value and gradually decrease to the desired value.
	*/
	DREAL relative_duality_gap=1.0;
	QPSolverTolRel = TolRel*0.1;

	init_time = CTime::get_curtime();
	total_time = CTime::get_curtime();
	nData = data_X->get_num_vectors();
	nDim = data_X->get_num_features();

	SG_PRINT("nDim=%d\nnData=%d\nC=%f\nBufSize=%d\nTolRel=%f\nTolAbs=%f\nMaxIter=%d\nThreads=%d\n\n",
			nDim,nData,C,BufSize,TolRel,TolAbs,MaxIter, num_threads);


	/* learned weight vector */
	//plhs[0] = (mxArray*)mxCreateDoubleMatrix(nDim,1,mxREAL);
	//W = (DREAL*)mxGetPr(plhs[0]);
	norm_W2 = 0;

	/* array of hinge poINTs used in line-serach  */
	hpf = (DREAL*) calloc(nData, sizeof(hpf[0]));
	if(hpf == NULL) SG_ERROR("Not enough memory for array hpf.");

	hpi = (INT*) calloc(nData, sizeof(hpi[0]));
	if(hpi == NULL) SG_ERROR("Not enough memory for array hpi.");

	/* previous value of the weight vector */
	Wold = (DREAL*)calloc(nDim,sizeof(DREAL));
	if(Wold == NULL) SG_ERROR("Not enough memory for vector Wold.");

	/* used to store dot products X'*W */
	proj = (DREAL*)calloc(nData,sizeof(DREAL));
	if(proj == NULL) SG_ERROR("Not enough memory for vector proj.");

	/* used to store dot products X'*Wold */
	proj_old = (DREAL*)calloc(nData,sizeof(DREAL));
	if(proj_old == NULL) SG_ERROR("Not enough memory for vector proj_old.");

	/* vectors Ci, Bi are used in the line search procedure */
	Ci = (DREAL*)calloc(nData,sizeof(DREAL));
	if(Ci == NULL) SG_ERROR("Not enough memory for vector Ci.");

	Bi = (DREAL*)calloc(nData,sizeof(DREAL));
	if(Bi == NULL) SG_ERROR("Not enough memory for vector Bi.");

	/* Hessian matrix contains dot product of normal vectors of selected cutting planes */
	H = (DREAL*)calloc(BufSize*BufSize,sizeof(DREAL));
	if(H == NULL) SG_ERROR("Not enough memory for matrix H.");

	/* bias of cutting planes */
	b = (DREAL*)calloc(BufSize,sizeof(DREAL));
	if(b == NULL) SG_ERROR("Not enough memory for vector b.");

	/*FIXME
	  A_len = calloc(BufSize,sizeof(INT));
	  A_idx = calloc(BufSize,sizeof(A_idx[0]));
	  A_val = calloc(BufSize,sizeof(A_val[0]));
	  if(A_len == NULL || A_idx==NULL || A_val==NULL)
	  SG_ERROR("Not enough memory for vector A_len, A_idx, A_val.");
	  */

	thread_slices = (INT*)calloc(num_threads,sizeof(INT));
	if(thread_slices == NULL) SG_ERROR("Not enough memory for vector num_threads.");

	thread_times = (DREAL*)calloc(num_threads,sizeof(DREAL));
	if(thread_times == NULL) SG_ERROR("Not enough memory for vector num_threads.");

	threads = (pthread_t*)calloc(num_threads,sizeof(pthread_t));
	if(threads== NULL) SG_ERROR("Not enough memory for threads structure.");

	params_output = (struct thread_params_output*)calloc(num_threads,sizeof(struct thread_params_output));
	if(params_output== NULL) SG_ERROR("Not enough memory for params structure.");

	params_add = (struct thread_params_add*)calloc(num_threads,sizeof(struct thread_params_add));
	if(params_add== NULL) SG_ERROR("Not enough memory for params structure.");

	alpha = (DREAL*)calloc(BufSize,sizeof(DREAL));
	if(alpha == NULL) SG_ERROR("Not enough memory for vector alpha.");

	new_a = (DREAL*)calloc(num_threads*nDim,sizeof(DREAL));
	if(new_a == NULL) SG_ERROR("Not enough memory for vector new_a.");

	I = (WORD*)calloc(BufSize,sizeof(WORD));
	if(I == NULL) SG_ERROR("Not enough memory for vector I.");
	for(i=0; i< BufSize; i++) I[i] = 1;

	diag_H = (DREAL*)calloc(BufSize,sizeof(DREAL));
	if(diag_H == NULL) SG_ERROR("Not enough memory for vector diag_H.");

	add_time = 0;
	sort_time = 0;
	solver_time = 0;
	wait_time = 0;
	output_time = 0;

	for (i=0; i<num_threads; i++)
	{
		thread_times[i]=0;
		thread_slices[i]=0;
	}

	nSel = 0;
	exitFlag = 0;
	nIter = 0;

	/* Computed the initial cutting plane and value of Q_P. */
	time_diff=CTime::get_curtime();
	xi = nData;
	err = nData;
	new_b = nData;


	/*for i=1:nData,
	  X(:,i) = X(:,i)*y(i);
	  end*/

	LONG nnz_split = data_X->get_num_nonzero_entries()/num_threads;

	/*FIXME
	for(i=0; i < nData; i++)
		mul_sparse_col(data_y[i], data_X, i);
	*/

	INT accum_nnz = 0;
	INT thr = 0;

	for(i=0; i < nData; i++)
	{
		nItems = data_X->get_num_sparse_vec_features(i);

		if (accum_nnz < nnz_split*(thr+1))
			accum_nnz+=nItems;
		else
		{
			thread_slices[thr]=i;
			SG_PRINT("slice[%d]=%d (split=%d)\n", thr, thread_slices[thr], nnz_split);
			accum_nnz+=nItems;
			thr++;
		}
	}

	/*  new_a = new_a + data_X(:,i) */
	for(i=0; i < nData; i++)
		data_X->add_to_dense_vec(1.0, i, new_a, data_X->get_num_features());

	Q_P = 0.5*norm_W2 + C*xi;
	Q_D = 0;
	init_time=CTime::get_curtime()-init_time;
	relative_duality_gap=(Q_P-Q_D)/ABS(Q_P);
	SG_PRINT("%4d: nSel=%4d, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f, xi=%f, err=%f\n",
			nIter,nSel,Q_P,Q_D,Q_P-Q_D,relative_duality_gap,xi, err/nData);

	/* main loop */
	while(exitFlag == 0 && nIter < MaxIter)
	{
		nIter++;

		/* append new cutting plane to A, b and update H */
		b[nSel] = -new_b;

		/*norm_a2 = 0;
		  ptr = nSel*nDim;
		  for(j=0; j < nDim; j++) {
		  A[ptr++] = new_a[j];
		  norm_a2 += new_a[j]*new_a[j];
		  }*/

		nItems = 0;
		norm_a2 = 0;
		for(j=0; j < nDim; j++ ) {
			if(new_a[j] != 0) {
				nItems++;
				norm_a2 += new_a[j]*new_a[j];
			}
		}

		/* A(:,nSel) = new_a, where A is parse */
		/*FIXME
		A_len[nSel] = nItems;
		if(nItems > 0) {
			A_idx[nSel] = calloc(nItems,sizeof(uINT32_T));
			A_val[nSel] = calloc(nItems,sizeof(DREAL));
			if(A_idx[nSel]==NULL || A_val[nSel]==NULL)
				SG_ERROR("Not enough memory for vector A_idx[nSel], A_val[nSel].");

			ptr = 0;
			for(j=0; j < nDim; j++ ) {
				if(new_a[j] != 0) {
					A_idx[nSel][ptr] = j;
					A_val[nSel][ptr++] = new_a[j];
				}
			}
		}
		*/

		/* alpha = [alpha;0];

		   % H = Z(:,1:nSel)'*Z(:,1:nSel);
		   tmp = full(Z(:,1:nSel-1)'*Z(:,nSel));
		   H(1:nSel-1,nSel) = tmp;
		   H(nSel,1:nSel-1) = tmp';
		   H(nSel,nSel) = full(Z(:,nSel)'*Z(:,nSel));
		   */

		/*FIXME
		H[INDEX2(nSel,nSel,BufSize)] = norm_a2;
		diag_H[nSel] = norm_a2;
		for(i=0; i < nSel; i++) {
			aa = 0;
			for(j=0; j < A_len[i]; j++) {
				aa += new_a[A_idx[i][j]]*A_val[i][j];
			}

			H[INDEX2(i,nSel,BufSize)] = aa;
			H[INDEX2(nSel,i,BufSize)] = aa;
		}
		*/

		nSel++;

		/* solve QP */

		time_diff = CTime::get_curtime();
		CQPSSVMLib solver(H, I, alpha, nSel, BufSize);
		solver_flag=solver.solve_qp(alpha, nSel);
		//solver_flag=solver.solve_qp( &get_col, diag_H, b, C, I, alpha,
		//		nSel, qpsolver_maxit, 0.0, QPSolverTolRel, &Q_D, &dummy, 0l );
		Q_D = -solver.get_dual();
		solver_time+=CTime::get_curtime()-time_diff;

		/* Wold=W; W = A(:,1:nSel)*alpha(1:nSel) */
		for(j=0; j < nDim; j++)
		{
			Wold[j] = W[j];
			W[j] = 0;
		}

		for(i=0; i < nSel; i++)
		{
			nItems = A_len[i];
			if(nItems > 0 && alpha[i] > 0)
			{
				for(j=0; j < nItems; j++)
					W[A_idx[i][j]] += alpha[i]*A_val[i][j];
			}
		}

		/* norm_W2 = W'*W */
		norm_W2 = 0;
		for(j=0; j < nDim; j++)
			norm_W2 += W[j]*W[j];

		/* select a new cutting plane */
		switch( method )
		{
			case E_SVMPERF:

				xi = 0;
				new_b = 0;
				for(j=0; j < nDim; j++) new_a[j] = 0;
				for(i=0; i < nData; i++)
				{
					/* wx=1 - W'*data_X(:,i) */
					wx = data_X->dense_dot(-1.0, i, W, data_X->get_num_features(), 1.0);

					if(wx >= 0) {
						xi += wx;
						new_b++;
						data_X->add_to_dense_vec(1.0, i, new_a, data_X->get_num_features());  /* new_a = new_a + data_X(:,i)  */
					}
				}
				Q_P = 0.5*norm_W2 + C*xi;

				SG_PRINT("%4d: nSel=%4d, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f, xi=%f, qp_exitflag=%d\n",
						nIter,nSel,Q_P,Q_D,Q_P-Q_D,(Q_P-Q_D)/ABS(Q_P),xi, solver_flag);

				break;

			case E_SVMOCAS:
				/* proj_old = proj;
				   proj = X'*W;
				   Ci = C*(1-proj_old);
				   Bi = C*(proj_old - proj);
				   */
				{
					time_diff = CTime::get_curtime();

					INT nt;
					INT nthreads=num_threads-1;
					INT end=0;
					INT step= nData/num_threads;

					if (step<1)
					{
						nthreads=nData-1;
						step=1;
					}

					for (nt=0; nt<nthreads; nt++)
					{
						params_output[nt].thread_time=&thread_times[nt];
						params_output[nt].C=C;
						params_output[nt].proj=proj;
						params_output[nt].proj_old=proj_old;
						params_output[nt].Ci=Ci;
						params_output[nt].Bi=Bi;
						params_output[nt].W=W;
						params_output[nt].data_X=data_X;
						if (nt==0)
							params_output[nt].start = 0;
						else
							params_output[nt].start = thread_slices[nt-1];
						params_output[nt].end = thread_slices[nt];

						if (pthread_create(&threads[nt], NULL, compute_output_helper, (void*)&params_output[nt]) != 0)
						{
							nthreads=nt;
							SG_WARNING("thread creation failed\n");
							break;
						}
						end=params_output[nt].end;
					}

					struct thread_params_output last_params;
					last_params.thread_time=&thread_times[num_threads-1];
					last_params.C=C;
					last_params.proj=proj;
					last_params.proj_old=proj_old;
					last_params.Ci=Ci;
					last_params.Bi=Bi;
					last_params.W=W;
					last_params.data_X=data_X;
					last_params.start = end;
					last_params.end = nData;

					compute_output_helper(&last_params);

					DREAL wtime=CTime::get_curtime();
					for (nt=0; nt<nthreads; nt++)
					{
						if (pthread_join(threads[nt], NULL) != 0)
							SG_WARNING( "pthread_join failed\n");
					}
					wait_time+=CTime::get_curtime()-wtime;
					output_time+=CTime::get_curtime()-time_diff;
				}

				/* A0 = (W-Wold)'*(W-Wold);
				   B0 = Wold'*(W-Wold); */
				A0 = 0; B0 = 0;
				for(j=0; j < nDim; j++) {
					A0 += (W[j]-Wold[j])*(W[j]-Wold[j]);
					B0 += Wold[j]*(W[j]-Wold[j]);
				}

				/* hp = inf*ones(nData,1);
				   idx = find(Bi~=0);
				   hp(idx) = -Ci(idx)./Bi(idx);
				   [hp_sort,idx_sort] = sort(hp);
				   Bsum = B0+sum(Bi(find(Bi < 0)));
				   */

				INT num_hp=0;
				Bsum = B0;
				for(i=0; i< nData; i++) {
					DREAL val;
					if(Bi[i] != 0)
						val = -Ci[i]/Bi[i];
					else
						val = CMath::INFTY;

					if (val>0)
					{
						hpi[num_hp] = i;
						hpf[num_hp] = val;
						num_hp++;
					}
					else
						Bsum+= ABS(Bi[i]);

					if(Bi[i] < 0)
						Bsum += Bi[i];
				}

				time_diff=CTime::get_curtime();
				INT qsort_threads=0;
				thread_qsort qthr;
				qthr.output=hpf;
				qthr.index=hpi;
				qthr.size=num_hp;
				qthr.qsort_threads=&qsort_threads;
				qthr.sort_limit=4096;;
				CMath::parallel_qsort_index<DREAL,INT>((void*) &qthr);
				sort_time+=CTime::get_curtime()-time_diff;

				/* t = hp_sort(1)-1;
				   i = 1;
				   Bsum = B0+sum(Bi(find(Bi < 0)));
				   GradVal = t*A0 + Bsum;
				   while GradVal < 0 & i <= nData & hp_sort(i) < inf,
				   t = hp_sort(i);
				   Bsum = Bsum + abs(Bi(idx_sort(i)));
				   GradVal = t*A0 + Bsum;
				   i = i+1;
				   end
				   */

				t = hpf[0] - 1;
				i = 0;
				GradVal = t*A0 + Bsum;
				while( GradVal < 0 && i < num_hp && hpf[i] < CMath::INFTY ) {
					t = hpf[i];
					Bsum = Bsum + ABS(Bi[hpi[i]]);
					GradVal = t*A0 + Bsum;
					i++;
				}

				/*
				   t = min(max(t+(1-t)/10,0.001),1);
				   W = Wold*(1-t) + t*W;
				   proj = proj_old*(1-t)+t*proj;

				   idx = find(proj >= 0);
				   new_z = sum(X(:,idx),2);
				   new_b = length(idx);

				   Q_P = 0.5*norm(W)^2 + C*sum(proj(idx));
				   */

				/*         t1 = MIN(MAX(t, 0.001),1.0); */             /* new W */
				/*         t2 = MIN(t+(1.0-t)/10.0,1.0);*/             /* nex cutting plane */

				t1 = t;                       /* new W */
				t2 = MIN(t+(1.0-t)/10.0,1.0); /* new cutting plane */


				norm_W2 = 0;
				for(j=0; j <nDim; j++) {
					W[j] = Wold[j]*(1-t1) + t1*W[j];
					norm_W2 += W[j]*W[j];
					new_a[j] = 0;
				}

				time_diff=CTime::get_curtime();
				err = 0;
				xi = 0;
				new_b = 0;

				{
					INT nt;
					INT nthreads=num_threads-1;
					INT end=0;
					INT step= nData/num_threads;

					if (step<1)
					{
						nthreads=nData-1;
						step=1;
					}

					for (nt=0; nt<nthreads; nt++)
					{
						DREAL* a=&new_a[nDim*(nt+1)];
						memset(a, 0, sizeof(DREAL)*nDim);

						params_add[nt].t1=t1;
						params_add[nt].t2=t2;
						params_add[nt].xi=xi;
						params_add[nt].err=err;
						params_add[nt].new_b=new_b;
						params_add[nt].proj=proj;
						params_add[nt].proj_old=proj_old;
						params_add[nt].data_X=data_X;
						params_add[nt].new_a=a;
						if (nt==0)
							params_add[nt].start = 0;
						else
							params_add[nt].start = thread_slices[nt-1];
						params_add[nt].end = thread_slices[nt];

						if (pthread_create(&threads[nt], NULL, add_helper, (void*)&params_add[nt]) != 0)
						{
							nthreads=nt;
							SG_WARNING("thread creation failed\n");
							break;
						}
						end=params_add[nt].end;
					}

					struct thread_params_add last_params;
					last_params.t1=t1;
					last_params.t2=t2;
					last_params.xi=xi;
					last_params.err=err;
					last_params.new_b=new_b;
					last_params.proj=proj;
					last_params.proj_old=proj_old;
					last_params.data_X=data_X;
					last_params.new_a=new_a;
					last_params.start = end;
					last_params.end = nData;
					add_helper(&last_params);

					xi+=last_params.xi;
					err+=last_params.err;
					new_b+=last_params.new_b;

					for (nt=0; nt<nthreads; nt++)
					{
						if (pthread_join(threads[nt], NULL) != 0)
							SG_WARNING( "pthread_join failed\n");

						xi+=params_add[nt].xi;
						err+=params_add[nt].err;
						new_b+=params_add[nt].new_b;
						DREAL* a=&new_a[nDim*(nt+1)];

						for (i=0; i<nDim; i++)
							new_a[i]+=a[i];
					}
				}
				add_time+=CTime::get_curtime()-time_diff;

				Q_P = 0.5*norm_W2 + C*xi;

				SG_PRINT("%4d: nSel=%4d, Q_P=%f, Q_D=%f, Q_P-Q_D=%f, Q_P-Q_D/abs(Q_P)=%f, xi=%f, err=%f, qp_exitflag=%d\n",
						nIter,nSel,Q_P,Q_D,Q_P-Q_D,(Q_P-Q_D)/ABS(Q_P),xi, err/nData, solver_flag);

				break;
		}

		if( Q_P - Q_D <= TolRel*ABS(Q_P)) exitFlag = 1;
		if( Q_P <= QPBound) exitFlag = 2;

		if(nSel >= BufSize) {
			SG_ERROR("Number of cutting planes exceeded BufSize (increase BufSize).");
			exitFlag = -1;
		}


	}

	SG_PRINT("exitflag=%d\n\n", exitFlag);

	total_time=CTime::get_curtime()-total_time;
	SG_PRINT("timing statistics\n"
			"===================\n"
			"init_time:%f\n"
			"solver_time: %f\n"
			"sort_time:%f\n"
			"add_time:%f\n"
			"thread_wait_time:%f\n",
			init_time, solver_time, sort_time, add_time,
			wait_time);

	for (i=0; i<num_threads; i++)
		SG_PRINT("thread_time[%d]=%fs\n", i, thread_times[i]);

	SG_PRINT("total_threads_output_time: %f\n"
			"total_time: %f\n\n", output_time, total_time);

	free(hpf);
	free(hpi);
	free(Wold);
	free(proj);
	free(proj_old);
	free(Bi);
	free(Ci);
	free(H);
	free(b);
	free(A_len);
	free(A_idx);
	free(A_val);
	free(thread_times);
	free(params_output);
	free(params_add);
	free(alpha);
	free(new_a);
	free(I);
	free(diag_H);
	free(thread_slices);

	return true;
}
