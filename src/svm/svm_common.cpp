/************************************************************************/
/*                                                                      */
/*   svm_common.c                                                       */
/*                                                                      */
/*   Definitions and functions used in both svm_learn and svm_classify. */
/*                                                                      */
/*   Author: Thorsten Joachims                                          */
/*   Date: 16.11.99                                                     */
/*                                                                      */
/*   Copyright (c) 1999  Universitaet Dortmund - All rights reserved    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include "svm/svm_common.h"
#include "svm/kernel.h"         /* this contains a user supplied kernel */

long   kernel_cache_statistic;

double classify_example(MODEL *model,DOC *ex) /* classifies example */
{
  register long i;
  register double dist;

  dist=0;
  for(i=1;i<model->sv_num;i++) {  
    dist+=kernel(&model->kernel_parm,model->supvec[i],ex)*model->alpha[i];
  }
  return(dist-model->b);
}

/* calculate the kernel function */
CFLOAT kernel(KERNEL_PARM *kernel_parm,DOC* a,DOC* b)
{
  kernel_cache_statistic++;
  switch(kernel_parm->kernel_type) {
////    case 0: /* linear */ 
////            return((CFLOAT)sprod_ss(a->words,b->words)); 
////    case 1: /* polynomial */
////            return((CFLOAT)pow(kernel_parm->coef_lin*sprod_ss(a->words,b->words)+kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
////    case 2: /* radial basis function */
////            return((CFLOAT)exp(-kernel_parm->rbf_gamma*(a->twonorm_sq-2*sprod_ss(a->words,b->words)+b->twonorm_sq)));
////    case 3: /* sigmoid neural net */
////            return((CFLOAT)tanh(kernel_parm->coef_lin*sprod_ss(a->words,b->words)+kernel_parm->coef_const)); 
		case 4: /* TOP Kernel */
	        return((CFLOAT)top_kernel(kernel_parm,a,b)); 
		case 5: 
			return ((CFLOAT)linear_top_kernel(kernel_parm,a,b)); 
	    default: printf("Error: Unknown kernel function\n"); exit(1);
	}
}

/* compute length of weight vector */
double model_length_s(MODEL *model, KERNEL_PARM *kernel_parm)
{
  register long i,j;
  register double sum=0,alphai;
  register DOC *supveci;

  for(i=1;i<model->sv_num;i++) {  
    alphai=model->alpha[i];
    supveci=model->supvec[i];
    for(j=1;j<model->sv_num;j++) {
      sum+=alphai*model->alpha[j]
	   *kernel(kernel_parm,supveci,model->supvec[j]);
    }
  }
  return(sqrt(sum));
}

void clear_vector_n(double *vec, long n)
{
  register long i;
  for(i=0;i<=n;i++) vec[i]=0;
}


void read_model(char *modelfile, MODEL *model,long max_words, long ll)
{
  FILE* modelfl;
  long j,i;
  char *line;
////  WORD *words;
  register long wpos;
  long wnum,pos;
  double weight;
  char version_buffer[100];

  if(verbosity>=1) {
    printf("Reading model..."); fflush(stdout);
  }
////  words = (WORD *)my_malloc(sizeof(WORD)*(max_words+10));
  line = (char *)my_malloc(sizeof(char)*ll);

  if ((modelfl = fopen (modelfile, "r")) == NULL)
  { perror (modelfile); exit (1); }

  fscanf(modelfl,"SVM-light Version %s\n",version_buffer);
  if(strcmp(version_buffer,VERSION)) {
    perror ("Version of model-file does not match version of svm_classify!"); 
    exit (1); 
  }
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.kernel_type);  
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.poly_degree);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.rbf_gamma);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_lin);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_const);
  fscanf(modelfl,"%s%*[^\n]\n", model->kernel_parm.custom);

////  fscanf(modelfl,"%ld%*[^\n]\n", &model->totwords);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->totdoc);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->sv_num);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->b);

  for(i=1;i<model->sv_num;i++) {
    fgets(line,(int)ll,modelfl);
    pos=0;
    wpos=0;
    sscanf(line,"%lf",&model->alpha[i]);
    while(line[++pos]>' ');
    while((sscanf(line+pos,"%ld:%lf",&wnum,&weight) != EOF) 
	  && (wpos<max_words)) {
      while(line[++pos]>' ');
////      words[wpos].wnum=wnum;
////      words[wpos].weight=weight; 
      wpos++;
    } 
    model->supvec[i] = (DOC *)my_malloc(sizeof(DOC));
////    (model->supvec[i])->words = (WORD *)my_malloc(sizeof(WORD)*(wpos+1));
    for(j=0;j<wpos;j++) {
////      (model->supvec[i])->words[j]=words[j]; 
    }
////    ((model->supvec[i])->words[wpos]).wnum=0;
////    (model->supvec[i])->twonorm_sq = sprod_ss((model->supvec[i])->words,
////					      (model->supvec[i])->words);
    (model->supvec[i])->docnum = -1;
  }

  fclose(modelfl);
  free(line);
	////  free(words);
  if(verbosity>=1) {
    fprintf(stdout, "OK. (%d support vectors read)\n",(int)(model->sv_num-1));
  }
}


long minl(long a, long b)
{
  if(a<b)
    return(a);
  else
    return(b);
}

long maxl(long a, long b)
{
  if(a>b)
    return(a);
  else
    return(b);
}

long get_runtime() 
{
  clock_t start;
  start = clock();
  return((long)((double)start*100.0/(double)CLOCKS_PER_SEC));
}


# ifdef _WIN32

int isnan(double a)
{
  return(_isnan(a));
}

# endif


void *my_malloc(long size)
{
  void *ptr;
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}

void copyright_notice()
{
  printf("\nCopyright: Thorsten Joachims, thorsten@ls8.cs.uni-dortmund.de\n\n");
  printf("This software is available for non-commercial use only. It must not\n");
  printf("be modified and distributed without prior permission of the author.\n");
  printf("The author is not responsible for implications from the use of this\n");
  printf("software.\n\n");
}
