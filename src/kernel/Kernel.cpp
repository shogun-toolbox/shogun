#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

/* calculate the kernel function */
REAL CKernel::kernel(CFeatures* a, int idx_a, CFeatures* b, int idx_b)
{
    if (idx_a < 0 || idx_b <0)
    {
#ifdef DEBUG
	printf("ERROR: (%d,%d)\n", a->docnum, b->docnum);
#endif
	return 0;
    }

	return compute(a,idx_a, b,idx_b);
}

/****************************** Cache handling *******************************/
/*
void CKernel::kernel_cache_init(KERNEL_CACHE *kernel_cache, long totdoc, long buffsize)
{
  long i;

  kernel_cache->index = new long[totdoc];
  kernel_cache->occu = new long[totdoc];
  kernel_cache->lru = new long[totdoc];
  kernel_cache->invindex = new long[totdoc];
  kernel_cache->active2totdoc = new long[totdoc];
  kernel_cache->totdoc2active = new long[totdoc];
  kernel_cache->buffer = new REAL[buffsize*1024*1024/sizeof(REAL)];

  kernel_cache->buffsize=(long)(buffsize*1024*1024/sizeof(REAL));

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/totdoc);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  if(verbosity>=2) {
   CIO::message(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
   CIO::message(" Kernel evals so far: %ld\n",kernel_cache_statistic);    
  }

  kernel_cache->elems=0;   // initialize cache 
  for(i=0;i<totdoc;i++) {
    kernel_cache->index[i]=-1;
    kernel_cache->lru[i]=0;
  }
  for(i=0;i<kernel_cache->max_elems;i++) {
    kernel_cache->occu[i]=0;
    kernel_cache->invindex[i]=-1;
  }

  kernel_cache->activenum=totdoc;;
  for(i=0;i<totdoc;i++) {
      kernel_cache->active2totdoc[i]=i;
      kernel_cache->totdoc2active[i]=i;
  }

  kernel_cache->time=0;  
} 

void CKernel::get_kernel_row(
KERNEL_CACHE *kernel_cache,
DOC *docs,          // Get's a row of the matrix of kernel values
long docnum,long totdoc, // This matrix has the same form as the Hessian,
long *active2dnum,  // just that the elements are not multiplied by
REAL *buffer,     // y_i * y_j * a_i * a_j
KERNEL_PARM *kernel_parm) // Takes the values from the cache if available.
{
  register long i,j,start;
  DOC *ex;

  ex=&(docs[docnum]);
  if(kernel_cache->index[docnum] != -1) { //is cached? 
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; // lru
    start=kernel_cache->activenum*kernel_cache->index[docnum];
    for(i=0;(j=active2dnum[i])>=0;i++) {
      if(kernel_cache->totdoc2active[j] >= 0) {
	buffer[j]=kernel_cache->buffer[start+kernel_cache->totdoc2active[j]];
      }
      else {
	buffer[j]=(REAL)kernel(kernel_parm,ex,&(docs[j]));
      }
    }
  }
  else {
    for(i=0;(j=active2dnum[i])>=0;i++) {
      buffer[j]=(REAL)kernel(kernel_parm,ex,&(docs[j]));
    }
  }
}


// Fills cache for the row m
void CKernel::cache_kernel_row(KERNEL_CACHE *kernel_cache, DOC *docs, long m, KERNEL_PARM *kernel_parm)
{
  register DOC *ex;
  register long j,k,l;
  register REAL *cache;

  if(!kernel_cache_check(kernel_cache,m)) {  // not cached yet
    cache = kernel_cache_clean_and_malloc(kernel_cache,m);
    if(cache) {
      l=kernel_cache->totdoc2active[m];
      ex=&(docs[m]);
      for(j=0;j<kernel_cache->activenum;j++) {  // fill cache 
	k=kernel_cache->active2totdoc[j];
	if((kernel_cache->index[k] != -1) && (l != -1) && (k != m)) {
	  cache[j]=kernel_cache->buffer[kernel_cache->activenum
				       *kernel_cache->index[k]+l];
	}
	else {
	  cache[j]=kernel(kernel_parm,ex,&(docs[k]));
	} 
      }
    }
    else {
      perror("Error: Kernel cache full! => increase cache size");
    }
  }
}

// Fills cache for the rows in key 
void CKernel::cache_multiple_kernel_rows(KERNEL_CACHE *kernel_cache, DOC *docs, long *key, long varnum, KERNEL_PARM *kernel_parm)
{
  register long i;

  for(i=0;i<varnum;i++) {  // fill up kernel cache 
    cache_kernel_row(kernel_cache,docs,key[i],kernel_parm);
  }
}

// remove numshrink columns in the cache
// which correspond to examples marked
void CKernel::kernel_cache_shrink(KERNEL_CACHE *kernel_cache, long totdoc, long numshrink, long *after)
{                           
  register long i,j,jj,from=0,to=0,scount;     // 0 in after.
  long *keep;

  if(verbosity>=2) {
   CIO::message(" Reorganizing cache...");
  }

  keep=new long[totdoc];
  for(j=0;j<totdoc;j++) {
    keep[j]=1;
  }
  scount=0;
  for(jj=0;(jj<kernel_cache->activenum) && (scount<numshrink);jj++) {
    j=kernel_cache->active2totdoc[jj];
    if(!after[j]) {
      scount++;
      keep[j]=0;
    }
  }

  for(i=0;i<kernel_cache->max_elems;i++) {
    for(jj=0;jj<kernel_cache->activenum;jj++) {
      j=kernel_cache->active2totdoc[jj];
      if(!keep[j]) {
	from++;
      }
      else {
	kernel_cache->buffer[to]=kernel_cache->buffer[from];
	to++;
	from++;
      }
    }
  }

  kernel_cache->activenum=0;
  for(j=0;j<totdoc;j++) {
    if((keep[j]) && (kernel_cache->totdoc2active[j] != -1)) {
      kernel_cache->active2totdoc[kernel_cache->activenum]=j;
      kernel_cache->totdoc2active[j]=kernel_cache->activenum;
      kernel_cache->activenum++;
    }
    else {
      kernel_cache->totdoc2active[j]=-1;
    }
  }

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/kernel_cache->activenum);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  delete[] keep;

  if(verbosity>=2) {
   CIO::message("done.\n");
   CIO::message(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
  }
}


void CKernel::kernel_cache_reset_lru(KERNEL_CACHE *kernel_cache)
{
  long maxlru=0,k;
  
  for(k=0;k<kernel_cache->max_elems;k++) {
    if(maxlru < kernel_cache->lru[k]) 
      maxlru=kernel_cache->lru[k];
  }
  for(k=0;k<kernel_cache->max_elems;k++) {
      kernel_cache->lru[k]-=maxlru;
  }
}

void CKernel::kernel_cache_cleanup(KERNEL_CACHE *kernel_cache)
{
  delete[] kernel_cache->index;
  delete[] kernel_cache->occu;
  delete[] kernel_cache->lru;
  delete[] kernel_cache->invindex;
  delete[] kernel_cache->active2totdoc;
  delete[] kernel_cache->totdoc2active;
  delete[] kernel_cache->buffer;
}

long CKernel::kernel_cache_malloc(KERNEL_CACHE *kernel_cache)
{
  long i;

  if(kernel_cache->elems < kernel_cache->max_elems) {
    for(i=0;i<kernel_cache->max_elems;i++) {
      if(!kernel_cache->occu[i]) {
	kernel_cache->occu[i]=1;
	kernel_cache->elems++;
	return(i);
      }
    }
  }
  return(-1);
}

void CKernel::kernel_cache_free(KERNEL_CACHE *kernel_cache, long i)
{
  kernel_cache->occu[i]=0;
  kernel_cache->elems--;
}

// remove least recently used cache
// element
long CKernel::kernel_cache_free_lru(KERNEL_CACHE *kernel_cache)  
{                                     
  register long k,least_elem=-1,least_time;

  least_time=kernel_cache->time+1;
  for(k=0;k<kernel_cache->max_elems;k++) {
    if(kernel_cache->invindex[k] != -1) {
      if(kernel_cache->lru[k]<least_time) {
	least_time=kernel_cache->lru[k];
	least_elem=k;
      }
    }
  }
  if(least_elem != -1) {
    kernel_cache_free(kernel_cache,least_elem);
    kernel_cache->index[kernel_cache->invindex[least_elem]]=-1;
    kernel_cache->invindex[least_elem]=-1;
    return(1);
  }
  return(0);
}

// Get a free cache entry. In case cache is full, the lru
// element is removed.
REAL* CKernel::kernel_cache_clean_and_malloc(KERNEL_CACHE *kernel_cache, long docnum)
{             
  long result;
  if((result = kernel_cache_malloc(kernel_cache)) == -1) {
    if(kernel_cache_free_lru(kernel_cache)) {
      result = kernel_cache_malloc(kernel_cache);
    }
  }
  kernel_cache->index[docnum]=result;
  if(result == -1) {
    return(0);
  }
  kernel_cache->invindex[result]=docnum;
  kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; // lru
  return((REAL *)((long)kernel_cache->buffer
		    +(kernel_cache->activenum*sizeof(REAL)*
		      kernel_cache->index[docnum])));
}

// Update lru time to avoid removal from cache.
long CKernel::kernel_cache_touch(KERNEL_CACHE *kernel_cache, long docnum)
{
  if(kernel_cache && kernel_cache->index[docnum] != -1) {
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; // lru
    return(1);
  }
  return(0);
}
  
// Is that row cached?
long CKernel::kernel_cache_check(KERNEL_CACHE *kernel_cache, long docnum)
{
  return(kernel_cache->index[docnum] != -1);
}
  

void CKernel::tester(KERNEL_PARM *kernel_parm)
{
	DOC a;
	DOC b;

	double kern[139][139];
	int i;
	for (i=0; i<139; i++)
	{
		for (int j=i; j<139; j++)
		{
			a.docnum=i;
			b.docnum=j;
			double v1=kernel(kernel_parm,&a,&b);

			a.docnum=i;
			b.docnum=j;
			double v2=kernel(kernel_parm,&a,&b);

			if (v2!=v1)
				printf("ERRROOOOOORR (%d,%d) -> %e|%e\n",i,j,v1,v2);

			kern[i][j]=v1;
			kern[j][i]=v2;
		}
	}

	for (i=0; i<139; i++)
	{
		for (int j=i; j<139; j++)
		{
			a.docnum=i;
			b.docnum=j;
			double v1=kernel(kernel_parm,&a,&b);

			a.docnum=i;
			b.docnum=j;
			double v2=kernel(kernel_parm,&a,&b);

			if (v2!=v1 || kern[i][j]-v1 || kern[j][i]!=v2)
				printf("ERRROOOOOORR\n\t\t (%d,%d) -> %e|%e %e|%e\n",i,j,kern[i][j],v1,kern[j][i],v2);
		}
	}
}

double CKernel::find_normalizer(KERNEL_PARM *kernel_parm, int num)
{
	DOC a;
	double sum=0;
	normalizer=1.0;

// do not normalize since they are already normalized
//#ifndef NORMALIZE_TO_ONE
	for (int i=0; i<num; i++)
	{
		a.docnum=i;
		sum+=kernel(kernel_parm, &a, &a);
	}
	normalizer=sum/num;
//#endif
	CIO::message("kernel normalizer: %f\n", normalizer);
	return normalizer;
}

double CKernel::linear_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b) 
  // plug in your favorite kernel
{
	double result=0;

	int x=a->docnum;
	int y=b->docnum;

	//calculate TOP Kernel
	if (x>=0 && y>=0)
	{
		double posx=pos->linear_model_probability(x);
		double posy=pos->linear_model_probability(y);
		double negx=neg->linear_model_probability(x);
		double negy=neg->linear_model_probability(y);

		result=(posx-negx)*(posy-negy);

		T_OBSERVATIONS* obs_x=(pos->get_observations())->get_obs_vector(x);
		T_OBSERVATIONS* obs_y=(pos->get_observations())->get_obs_vector(y);

		for (int i=0; i<pos->get_N(); i++)
		{
		    if (*obs_x==*obs_y)
			result+= (exp(-pos->get_b(i, *obs_x)-pos->get_b(i, *obs_y))) + 
			         (exp(-neg->get_b(i, *obs_x)-neg->get_b(i, *obs_y)));
		    obs_x++;
		    obs_y++;
		}

	}
	return result/normalizer;
}


double CKernel::top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b)
{
    double result=0;

    int x=a->docnum;
    int y=b->docnum;

    int i,j,p=0;
    double posx=pos->model_probability(x);
    double negx=neg->model_probability(x);
    
    theta[p++]=(posx-negx);

    //first do all derivatives of parameters of positive model for sequence x
    for (i=0; i<pos->get_N(); i++)
    {
	theta[p++]=exp(pos->model_derivative_p(i, x)-posx);
	theta[p++]=exp(pos->model_derivative_q(i, x)-posx);

	for (j=0; j<pos->get_N(); j++)
	    theta[p++]=exp(pos->model_derivative_a(i, j, x)-posx);

	for (j=0; j<pos->get_M(); j++)
	    theta[p++]=exp(pos->model_derivative_b(i, j, x)-posx);

    }
    
    //then do all derivatives of parameters of negative model for sequence y
    for (i=0; i<neg->get_N(); i++)
    {
	theta[p++]=exp(neg->model_derivative_p(i, x)-negx);
	theta[p++]=exp(neg->model_derivative_q(i, x)-negx);

	for (j=0; j<neg->get_N(); j++)
	    theta[p++]=exp(neg->model_derivative_a(i, j, x)-negx);

	for (j=0; j<neg->get_M(); j++)
	    theta[p++]=exp(neg->model_derivative_b(i, j, x)-negx);

    }

    p=0;  
    double posy=pos->model_probability(y);
    double negy=neg->model_probability(y);

    result=theta[p++]*(posy-negy);

    //second do all derivatives of parameters of positive model for sequence y
    for (i=0; i<pos->get_N(); i++)
    {
	result+=theta[p++]*exp(pos->model_derivative_p(i, y)-posy);
	result+=theta[p++]*exp(pos->model_derivative_q(i, y)-posy);

	for (j=0; j<pos->get_N(); j++)
	    result+=theta[p++]*exp(pos->model_derivative_a(i, j, y)-posy);

	for (j=0; j<pos->get_M(); j++)
	    result+=theta[p++]*exp(pos->model_derivative_b(i, j, y)-posy);
    }

    //... and last derivatives of parameters of negative model for sequence y
    for (i=0; i<neg->get_N(); i++)
    {
	result+=theta[p++]*exp(neg->model_derivative_p(i, y)-negy);
	result+=theta[p++]*exp(neg->model_derivative_q(i, y)-negy);

	for (j=0; j<neg->get_N(); j++)
	    result+=theta[p++]*exp(neg->model_derivative_a(i, j, y)-negy);

	for (j=0; j<neg->get_M(); j++)
	    result+=theta[p++]*exp(neg->model_derivative_b(i, j, y)-negy);
    }

    return result/normalizer;
}

double CKernel::cached_top_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b)
{
    double* features_a=CHMM::get_top_feature_cache_line(a->docnum);
    double* features_b=CHMM::get_top_feature_cache_line(b->docnum);
    double result=0;
    int i=CHMM::get_top_num_features();

    while (i--)
	result+= *features_a++ * *features_b++;

    result/=normalizer;
#ifdef KERNEL_DEBUG
    double top_res=top_kernel(kernel_parm,a,b);
    if (fabs(top_res-result)>1e-6)
	printf("cached kernel bug:%e == %e\n", top_kernel(kernel_parm,a,b), result);
#endif
    return result;
}


double CKernel::cached_fisher_kernel(KERNEL_PARM *kernel_parm, DOC* a, DOC* b)
{
    double* features_a=CHMM::get_top_feature_cache_line(a->docnum);
    double* features_b=CHMM::get_top_feature_cache_line(b->docnum);
    double result=0;
    int i=CHMM::get_top_num_features();

    while (i--)
		result+= *features_a++ * *features_b++;

    features_a=CHMM::get_top_feature_cache_line(a->docnum);
    features_b=CHMM::get_top_feature_cache_line(b->docnum);
	//fisher kernel is just the same as TOP but without first komponent
	result-=features_a[0] * features_b[0];

    result/=normalizer;
#ifdef KERNEL_DEBUG
    double top_res=top_kernel(kernel_parm,a,b);
    if (fabs(top_res-result)>1e-6)
	printf("cached kernel bug:%e == %e\n", top_kernel(kernel_parm,a,b), result);
#endif
    return result;
}

bool CKernel::save_kernel(FILE* dest, CObservation* obs, int kernel_type)
{
    KERNEL_CACHE mykernel_cache;
    KERNEL_PARM mykernel_parm;
	int totdoc=obs->get_DIMENSION();
	
	memset(&mykernel_cache, 0x0, sizeof(KERNEL_CACHE));
	memset(&mykernel_parm, 0x0, sizeof(KERNEL_PARM));

	if (kernel_type==6)
	{
		if (!CHMM::compute_top_feature_cache(pos, neg))
			kernel_type=4; // hmm+svm precalculated
	}
	else if (kernel_type==7)
	{
		if (!CHMM::compute_top_feature_cache(pos, neg))
			CIO::message("preparing for crash...");
	}
	
	mykernel_parm.kernel_type=kernel_type; //custom kernel
	mykernel_parm.poly_degree=-12345;
	mykernel_parm.rbf_gamma=-12345;
	mykernel_parm.coef_lin=-12345;
	mykernel_parm.coef_const=-12345;
	
	find_normalizer(&mykernel_parm, totdoc);

#ifdef USE_KERNEL_CACHE
	kernel_cache_init(&mykernel_cache,totdoc,100);
#else
	kernel_cache_init(&mykernel_cache,totdoc, 2);
#endif

	DOC a;
	DOC b;
	for (int i=0; i<totdoc; i++)
	{
		a.docnum=i;
		for (int j=0; j<totdoc; j++)
		{
			b.docnum=j;
			double d=kernel(&mykernel_parm, &a, &b);
			fwrite(&d, sizeof(double),1, dest);
		}
	}

	kernel_cache_cleanup(&mykernel_cache);
	return true;
}*/

/*
init
  if(kernel_cache) {
    kernel_cache->time=iteration;  // for lru cache
    kernel_cache_reset_lru(kernel_cache);
  }


   kernel_cache->time=iteration;  // for lru cache 
    

      if((kernel_cache)
	 && (supvecnum>kernel_cache->max_elems)
	 && ((kernel_cache->activenum-activenum)>math.max((long)(activenum/10),(long) 500))) {
	kernel_cache_shrink(kernel_cache,totdoc,math.max((long)(activenum/10),(long) 500),
			    shrink_state->active); 
      }
    }
if(kernel_cache) 
      cache_multiple_kernel_rows(kernel_cache,docs,working2dnum,
				 choosenum,kernel_parm); 
    
*/
