#include "lib/io.h"
#include "svm/SVM_light.h"

extern long verbosity;
extern double normalizer;
extern CHMM* pos;
extern CHMM* neg;

CSVMLight::CSVMLight()
{
  mymodel.supvec=NULL ;
  mymodel.alpha=NULL ;
  mymodel.index=NULL ;
}

CSVMLight::~CSVMLight()
{
  delete[] mymodel.supvec;
  delete[] mymodel.alpha;
  delete[] mymodel.index ;
}

bool CSVMLight::svm_train(CObservation* train, int kernel_type, double C)
{
	delete[] docs;
	delete[] label;
	docs=NULL;
	label=NULL;
	free(mymodel.supvec);
	free(mymodel.alpha);
	free(mymodel.index);
	memset(&mymodel, 0x0, sizeof(MODEL));
	memset(&mymodel, 0x0, sizeof(MODEL));
	memset(&mykernel_cache, 0x0, sizeof(KERNEL_CACHE));
	memset(&mylearn_parm, 0x0, sizeof(LEARN_PARM));
	memset(&mykernel_parm, 0x0, sizeof(KERNEL_PARM));

	int totdoc=train->get_DIMENSION();

	docs=new DOC[totdoc];  /* training examples */
	label=new long[totdoc];
	char str[1024];

	if (kernel_type==4) // standard hmm+svm
	{
	    if (CHMM::compute_top_feature_cache(pos, neg))
		kernel_type=6; // hmm+svm precalculated
	}

	for (int i=0; i<totdoc; i++)
	{
		docs[i].docnum=i;
		docs[i].twonorm_sq=0;
		label[i]=train->get_label(i);
		//printf("%i -> %i\n", (int) docs[i].docnum, (int) label[i]);
	}

	verbosity=1;
	strcpy (mylearn_parm.predfile, "");
	strcpy (mylearn_parm.alphafile, "");
	mylearn_parm.biased_hyperplane=1;
	mylearn_parm.remove_inconsistent=0;
	mylearn_parm.skip_final_opt_check=1;
	mylearn_parm.svm_maxqpsize=50;
	mylearn_parm.svm_newvarsinqp=0;
	mylearn_parm.svm_iter_to_shrink=100;
	mylearn_parm.svm_c=C;
	mylearn_parm.transduction_posratio=0.5;
	mylearn_parm.svm_costratio=1.0;
	mylearn_parm.svm_costratio_unlab=1.0;
	mylearn_parm.svm_unlabbound=1E-5;
	mylearn_parm.epsilon_crit=1E-6;
	mylearn_parm.epsilon_a=1E-15;
	mylearn_parm.compute_loo=0;
	mylearn_parm.rho=1.0;
	mylearn_parm.xa_depth=0;
	mykernel_parm.kernel_type=kernel_type; //custom kernel
	mykernel_parm.poly_degree=-12345;
	mykernel_parm.rbf_gamma=-12345;
	mykernel_parm.coef_lin=-12345;
	mykernel_parm.coef_const=-12345;
	
	//tester(&kernel_parm); //to check kernel

	double norm_val=find_normalizer(&mykernel_parm, totdoc);

	sprintf(str,"%.32g",norm_val);
	strcpy(mykernel_parm.custom, str);

#ifdef USE_KERNEL_CACHE
	kernel_cache_init(&mykernel_cache,totdoc,100);
	svm_learn(docs,label,totdoc,-12345,&mylearn_parm,&mykernel_parm,&mykernel_cache,&mymodel);
	kernel_cache_cleanup(&mykernel_cache);
#else
	kernel_cache_init(&mykernel_cache,totdoc, 2);
	svm_learn(docs,label,totdoc,-12345,&mylearn_parm,&mykernel_parm,&mykernel_cache,&mymodel);
	kernel_cache_cleanup(&mykernel_cache);
#endif
	if (kernel_type==6)
	    mymodel.kernel_parm.kernel_type=4;
	

	return true;
}

bool CSVMLight::svm_test(CObservation* test, FILE* outfile, FILE* rocfile)
{
    DOC doc;   // test example

    int total=test->get_DIMENSION();
    
    if (mymodel.kernel_parm.kernel_type==4) // standard hmm+svm
    {
	if (CHMM::compute_top_feature_cache(pos, neg))
	    mymodel.kernel_parm.kernel_type=6; // hmm+svm precalculated
    }

    double *output = new double[total];	
    int* label= new int[total];	

    for (int i=0; i<total; i++)
    { 
	doc.docnum=i;
	doc.twonorm_sq=-1;
	output[i]=classify_example(&mymodel,&doc);

	label[i]=test->get_label(i);
	if ((label[i] < 0 && output[i] < 0) || (label[i] > 0 && output[i] > 0))
	    CIO::message(outfile,"%+.8g (%+d)\n",output[i], label[i]);
	else
	    CIO::message(outfile,"%+.8g (%+d)(*)\n",output[i], label[i]);
    }  

    double *fp= new double[total];	
    double *tp= new double[total];	
    int possize=-1;
    int negsize=-1;

    int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize, rocfile);

    double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
    double fpo=fp[pointeven]*negsize;
    double fne=(1-tp[pointeven])*possize;

   CIO::message("classified:\n");
   CIO::message("\tcorrect:%i\n", int (correct));
   CIO::message("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
   CIO::message("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, fp[pointeven], tp[pointeven]);
    delete[] fp;
    delete[] tp;
    delete[] output;
    delete[] label;
    return true;

  /*while((!feof(docfl)) && fgets(line,(int)lld,docfl)) {
    if(line[0] == '#') continue;  // line contains comments
    parse_document(line,&doc,&doc_label,&wnum,max_words_doc);
    totdoc++;
    
      t1=get_runtime();
      dist=classify_example(&mymodel,&doc);
      runtime+=(get_runtime()-t1);
    
    if(dist>0) {
      if(pred_format==0) { // old weired output format
	fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
      }
      if(doc_label==1) correct++; else incorrect++;
      if(doc_label==1) res_a++; else res_b++;
    }
    else {
      if(pred_format==0) { // old weired output format
	fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
      }
      if(doc_label==-1) correct++; else incorrect++;
      if(doc_label==1) res_c++; else res_d++;
    }
    if(pred_format==1) { // output the value of decision function
      fprintf(predfl,"%.8g\n",dist);
    }
    if((doc_label*doc_label) != 1) { no_accuracy=1; }
    if(verbosity>=2) {
      if(totdoc % 100 == 0) {
	printf("%ld..",totdoc);
      }
    }
  }  
  free(line);
  free(doc.words);
  //  free(mymodel.supvec);
  //    free(mymodel.alpha);*/
}

bool CSVMLight::load_svm(FILE* modelfl, CObservation* test)
{
    bool result=false;
    char version_buffer[1024];

    memset(&mymodel, 0x0, sizeof(MODEL));
    fscanf(modelfl,"SVM-light Version %s\n",version_buffer);
    if(strcmp(version_buffer,VERSION)) {
	perror ("Version of model-file does not match version of svm_classify!"); 
	exit (1); 
    }
    fscanf(modelfl,"%ld%*[^\n]\n", &mymodel.kernel_parm.kernel_type);  
//    fscanf(modelfl,"%ld%*[^\n]\n", &model.kernel_parm.poly_degree);
//    fscanf(modelfl,"%lf%*[^\n]\n", &model.kernel_parm.rbf_gamma);
//    fscanf(modelfl,"%lf%*[^\n]\n", &model.kernel_parm.coef_lin);
//    fscanf(modelfl,"%lf%*[^\n]\n", &model.kernel_parm.coef_const);
    fscanf(modelfl,"%lf%*[^\n]\n", &normalizer);
#ifdef DEBUG
	printf("normalizer:%e\n",normalizer);
#endif

    ////  fscanf(modelfl,"%ld%*[^\n]\n", &mymodel.totwords);
    fscanf(modelfl,"%ld%*[^\n]\n", &mymodel.totdoc);
    fscanf(modelfl,"%ld%*[^\n]\n", &mymodel.sv_num);
    fscanf(modelfl,"%lf%*[^\n]\n", &mymodel.b);
    int file_pos=ftell(modelfl);
    
    mymodel.sv_num--;
   CIO::message("loading %ld support vectors\n",mymodel.sv_num);
    test->add_support_vectors(modelfl, mymodel.sv_num);
    fseek(modelfl, file_pos, SEEK_SET);

    mymodel.supvec=new DOC*[mymodel.sv_num];
    mymodel.alpha=new double[mymodel.sv_num];

    for (int i=0; i<mymodel.sv_num; i++)
    {
	mymodel.supvec[i] = new DOC;
	mymodel.supvec[i]->docnum=test->get_support_vector_idx(i);
	mymodel.supvec[i]->twonorm_sq=-1;
	fscanf(modelfl,"%lf%*[^\n]\n",&mymodel.alpha[i]);
#ifdef DEBUG
	printf("alpha:%e,idx:%d\n",mymodel.alpha[i],mymodel.supvec[i]->docnum);
#endif
    }

    result=true;
    svm_loaded=result;
    return result;
}

bool CSVMLight::save_svm(FILE* modelfl)
{
  write_model(modelfl,&mymodel);
  return true ;
} 

double CSVMLight::classify_example(MODEL *model,DOC *ex) /* classifies example */
{
  register long i;
  register double dist;

  dist=0;
  for(i=1;i<model->sv_num;i++) { 
//    CIO::message("(%d,%d) alpha:%f kernel:%f\n ", ex->docnum, model->supvec[i]->docnum, model->alpha[i], kernel(&model->kernel_parm,model->supvec[i],ex));
    dist+=kernel(&model->kernel_parm,model->supvec[i],ex)*model->alpha[i];
  }
  return(dist-model->b);
}

/* compute length of weight vector */
double CSVMLight::model_length_s(MODEL *model, KERNEL_PARM *kernel_parm)
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

void CSVMLight::clear_vector_n(double *vec, long n)
{
  register long i;
  for(i=0;i<=n;i++) vec[i]=0;
}


void CSVMLight::read_model(char *modelfile, MODEL *model,long max_words, long ll)
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
   CIO::message("Reading model...");
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
    CIO::message( "OK. (%d support vectors read)\n",(int)(model->sv_num-1));
  }
}

void* CSVMLight::my_malloc(long size)
{
  void *ptr;
  ptr=(void *)malloc(size);
  if(!ptr) { 
    perror ("Out of memory!\n"); 
    exit (1); 
  }
  return(ptr);
}

long CSVMLight::get_runtime() 
{
  clock_t start;
  start = clock();
  return((long)((double)start*100.0/(double)CLOCKS_PER_SEC));
}

void CSVMLight::svm_learn(
DOC *docs,                 /* Training vectors (x-part) */
long *label,               /* Training labels (y-part, zero if test example for transduction) */ 
long totdoc,               /* Number of examples in docs/label */
long totwords,             /* Number of features (i.e. highest feature 
			      index) */
LEARN_PARM *learn_parm,    /* Learning paramenters */
KERNEL_PARM *kernel_parm,  /* Kernel paramenters */
KERNEL_CACHE *kernel_cache,/* Initialed Cache */
MODEL *model)              /* Returns learning result (assumed empty before 
			      called) */
{
  long *inconsistent,i;
  long inconsistentnum;
  long misclassified,upsupvecnum;
  double loss,model_length,example_length;
  double maxdiff,*lin,*a;
  long runtime_start,runtime_end;
  long iterations;
  long *unlabeled,transduction;
  long heldout;
  long loo_count=0,loo_count_pos=0,loo_count_neg=0,trainpos=0,trainneg=0;
  long loocomputed=0,runtime_start_loo=0,runtime_start_xa=0;
  double heldout_c=0,r_delta_sq=0,r_delta,r_delta_avg;

  double *xi_fullset; /* buffer for storing xi on full sample in loo */
  double *a_fullset;  /* buffer for storing alpha on full sample in loo */
  TIMING timing_profile;
  SHRINK_STATE shrink_state;

  runtime_start=get_runtime();
  timing_profile.time_kernel=0;
  timing_profile.time_opti=0;
  timing_profile.time_shrink=0;
  timing_profile.time_update=0;
  timing_profile.time_model=0;
  timing_profile.time_check=0;
  timing_profile.time_select=0;
  kernel_cache_statistic=0;

////  learn_parm->totwords=totwords;

  /* make sure -n value is reasonable */
  if((learn_parm->svm_newvarsinqp < 2) 
     || (learn_parm->svm_newvarsinqp > learn_parm->svm_maxqpsize)) {
    learn_parm->svm_newvarsinqp=learn_parm->svm_maxqpsize;
  }

  init_shrink_state(&shrink_state,totdoc,(long)10000);

  inconsistent = (long *)my_malloc(sizeof(long)*totdoc);
  unlabeled = (long *)my_malloc(sizeof(long)*totdoc);
  a = (double *)my_malloc(sizeof(double)*totdoc);
  a_fullset = (double *)my_malloc(sizeof(double)*totdoc);
  xi_fullset = (double *)my_malloc(sizeof(double)*totdoc);
  lin = (double *)my_malloc(sizeof(double)*totdoc);
  learn_parm->svm_cost = (double *)my_malloc(sizeof(double)*totdoc);
  model->supvec = (DOC **)my_malloc(sizeof(DOC *)*(totdoc+2));
  model->alpha = (double *)my_malloc(sizeof(double)*(totdoc+2));
  model->index = (long *)my_malloc(sizeof(long)*(totdoc+2));

  model->at_upper_bound=0;
  model->b=0;	       
  model->supvec[0]=0;  /* element 0 reserved and empty for now */
  model->alpha[0]=0;
////  model->lin_weights=NULL;
////  model->totwords=totwords;
  model->totdoc=totdoc;
  model->kernel_parm=(*kernel_parm);
  model->sv_num=1;
  model->loo_error=-1;
  model->loo_recall=-1;
  model->loo_precision=-1;
  model->xa_error=-1;
  model->xa_recall=-1;
  model->xa_precision=-1;
  inconsistentnum=0;
  transduction=0;

  r_delta=estimate_r_delta(docs,totdoc,kernel_parm);
  r_delta_sq=r_delta*r_delta;

  r_delta_avg=estimate_r_delta_average(docs,totdoc,kernel_parm);
  if(learn_parm->svm_c == 0.0) {  /* default value for C */
    learn_parm->svm_c=1.0/(r_delta_avg*r_delta_avg);
    if(verbosity>=1) 
     CIO::message("Setting default regularization parameter C=%.4f\n",
	     learn_parm->svm_c);
  }

  for(i=0;i<totdoc;i++) {    /* various inits */
    inconsistent[i]=0;
    a[i]=0;
    lin[i]=0;
    unlabeled[i]=0;
    if(label[i] == 0) {
      unlabeled[i]=1;
      transduction=1;
    }
    if(label[i] > 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*learn_parm->svm_costratio*
	fabs((double)label[i]);
      label[i]=1;
      trainpos++;
    }
    else if(label[i] < 0) {
      learn_parm->svm_cost[i]=learn_parm->svm_c*fabs((double)label[i]);
      label[i]=-1;
      trainneg++;
    }
    else {
      learn_parm->svm_cost[i]=0;
    }
  }

  /* caching makes no sense for linear kernel */
////  if(kernel_parm->kernel_type == LINEAR) {
////    kernel_cache = NULL;   
////  } 

  if(transduction) {
    learn_parm->svm_iter_to_shrink=99999999;
    if(verbosity >= 1)
     CIO::message("\nDeactivating Shrinking due to an incompatibility with the transductive \nlearner in the current version.\n\n");
  }

  if(transduction && learn_parm->compute_loo) {
    learn_parm->compute_loo=0;
    if(verbosity >= 1)
     CIO::message("\nCannot compute leave-one-out estimates for transductive learner.\n\n");
  }    

  if(learn_parm->remove_inconsistent && learn_parm->compute_loo) {
    learn_parm->compute_loo=0;
   CIO::message("\nCannot compute leave-one-out estimates when removing inconsistent examples.\n\n");
  }    

  if((trainpos == 1) || (trainneg == 1)) {
    learn_parm->compute_loo=0;
   CIO::message("\nCannot compute leave-one-out with only one example in one class.\n\n");
  }    


  if(verbosity==1) {
   CIO::message("Optimizing");
  }

  /* train the svm */
  iterations=optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
				     kernel_parm,kernel_cache,&shrink_state,model,
				     inconsistent,unlabeled,a,lin,&timing_profile,
				     &maxdiff,(long)-1,
				     (long)1);
  
  if(verbosity>=1) {
    if(verbosity==1)CIO::message("done. (%ld iterations)\n",iterations);

    misclassified=0;
    for(i=0;(i<totdoc);i++) { /* get final statistic */
      if((lin[i]-model->b)*(double)label[i] <= 0.0) 
	misclassified++;
    }

   CIO::message("Optimization finished (%ld misclassified, maxdiff=%.5f).\n",
	   misclassified,maxdiff); 

    runtime_end=get_runtime();
    if(verbosity>=2) {
     CIO::message("Runtime in cpu-seconds: %.2f (%.2f%% for kernel/%.2f%% for optimizer/%.2f%% for final/%.2f%% for update/%.2f%% for model/%.2f%% for check/%.2f%% for select)\n",
        ((float)runtime_end-(float)runtime_start)/100.0,
        (100.0*timing_profile.time_kernel)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_opti)/(float)(runtime_end-runtime_start),
	(100.0*timing_profile.time_shrink)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_update)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_model)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_check)/(float)(runtime_end-runtime_start),
        (100.0*timing_profile.time_select)/(float)(runtime_end-runtime_start));
    }
    else {
     CIO::message("Runtime in cpu-seconds: %.2f\n",
	     (runtime_end-runtime_start)/100.0);
    }

    if(learn_parm->remove_inconsistent) {	  
      inconsistentnum=0;
      for(i=0;i<totdoc;i++) 
	if(inconsistent[i]) 
	  inconsistentnum++;
     CIO::message("Number of SV: %ld (plus %ld inconsistent examples)\n",
	     model->sv_num-1,inconsistentnum);
    }
    else {
      upsupvecnum=0;
      for(i=1;i<model->sv_num;i++) {
	if(fabs(model->alpha[i]) >= 
	   (learn_parm->svm_cost[(model->supvec[i])->docnum]-
	    learn_parm->epsilon_a)) 
	  upsupvecnum++;
      }
     CIO::message("Number of SV: %ld (including %ld at upper bound)\n",
	     model->sv_num-1,upsupvecnum);
    }
    
    if((verbosity>=1) && (!learn_parm->skip_final_opt_check)) {
      loss=0;
      model_length=0; 
      for(i=0;i<totdoc;i++) {
	if((lin[i]-model->b)*(double)label[i] < 1.0-learn_parm->epsilon_crit)
	  loss+=1.0-(lin[i]-model->b)*(double)label[i];
	model_length+=a[i]*label[i]*lin[i];
      }
      model_length=sqrt(model_length);
      CIO::message("L1 loss: loss=%.5f\n",loss);
      CIO::message("Norm of weight vector: |w|=%.5f\n",model_length);
      example_length=estimate_sphere(model,kernel_parm); 
      CIO::message("Norm of longest example vector: |x|=%.5f\n",
	      length_of_longest_document_vector(docs,totdoc,kernel_parm));
      CIO::message("Estimated VCdim of classifier: VCdim<=%.5f\n",
	      estimate_margin_vcdim(model,model_length,example_length,
				    kernel_parm));
      if((!learn_parm->remove_inconsistent) && (!transduction)) {
	runtime_start_xa=get_runtime();
	if(verbosity>=1) {
	 CIO::message("Computing XiAlpha-estimates...");
	}
	compute_xa_estimates(model,label,unlabeled,totdoc,docs,lin,a,
			     kernel_parm,learn_parm,&(model->xa_error),
			     &(model->xa_recall),&(model->xa_precision));
	if(verbosity>=1) {
	 CIO::message("done\n");
	}
	printf("Runtime for XiAlpha-estimates in cpu-seconds: %.2f\n",
	       (get_runtime()-runtime_start_xa)/100.0);
	
	CIO::message("XiAlpha-estimate of the error: error<=%.2f%% (rho=%.2f,depth=%ld)\n",
		model->xa_error,learn_parm->rho,learn_parm->xa_depth);
	CIO::message("XiAlpha-estimate of the recall: recall=>%.2f%% (rho=%.2f,depth=%ld)\n",
		model->xa_recall,learn_parm->rho,learn_parm->xa_depth);
	CIO::message("XiAlpha-estimate of the precision: precision=>%.2f%% (rho=%.2f,depth=%ld)\n",
		model->xa_precision,learn_parm->rho,learn_parm->xa_depth);
      }
      else if(!learn_parm->remove_inconsistent) {
	estimate_transduction_quality(model,label,unlabeled,totdoc,docs,lin);
      }
    }
    if(verbosity>=1) {
     CIO::message("Number of kernel evaluations: %ld\n",kernel_cache_statistic);
    }
  }


  /* leave-one-out testing starts now */
  if(learn_parm->compute_loo) {
    /* save results of training on full dataset for leave-one-out */
    runtime_start_loo=get_runtime();
    for(i=0;i<totdoc;i++) {
      xi_fullset[i]=1.0-((lin[i]-model->b)*(double)label[i]);
      a_fullset[i]=a[i];
    }
    if(verbosity>=1) {
     CIO::message("Computing leave-one-out");
    }
    
    /* repeat this loop for every held-out example */
    for(heldout=0;(heldout<totdoc);heldout++) {
      if(learn_parm->rho*a_fullset[heldout]*r_delta_sq+xi_fullset[heldout]
	 < 1.0) { 
	/* guaranteed to not produce a leave-one-out error */
	if(verbosity==1) {
	 CIO::message("+"); 
	}
      }
      else if(xi_fullset[heldout] > 1.0) {
	/* guaranteed to produce a leave-one-out error */
	loo_count++;
	if(label[heldout] > 0)  loo_count_pos++; else loo_count_neg++;
	if(verbosity==1) {
	 CIO::message("-"); 
	}
      }
      else {
	loocomputed++;
	heldout_c=learn_parm->svm_cost[heldout]; /* set upper bound to zero */
	learn_parm->svm_cost[heldout]=0;
	/* make sure heldout example is not currently  */
	/* shrunk away. Assumes that lin is up to date! */
	shrink_state.active[heldout]=1;  
	if(verbosity>=2) 
	 CIO::message("\nLeave-One-Out test on example %ld\n",heldout);
	if(verbosity>=1) {
	 CIO::message("(?[%ld]",heldout); 
	}
	
	optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
				kernel_parm,
				kernel_cache,&shrink_state,model,inconsistent,unlabeled,
				a,lin,&timing_profile,
				&maxdiff,heldout,(long)2);

	/*CIO::message("%f\n",(lin[heldout]-model->b)*(double)label[heldout]); */

	if(((lin[heldout]-model->b)*(double)label[heldout]) < 0.0) { 
	  loo_count++;                            /* there was a loo-error */
	  if(label[heldout] > 0)  loo_count_pos++; else loo_count_neg++;
	  if(verbosity>=1) {
	   CIO::message("-)"); 
	  }
	}
	else {
	  if(verbosity>=1) {
	   CIO::message("+)"); 
	  }
	}
	/* now we need to restore the original data set*/
	learn_parm->svm_cost[heldout]=heldout_c; /* restore upper bound */
      }
    } /* end of leave-one-out loop */


    if(verbosity>=1) {
     CIO::message("\nRetrain on full problem"); 
    }
    optimize_to_convergence(docs,label,totdoc,totwords,learn_parm,
			    kernel_parm,
			    kernel_cache,&shrink_state,model,inconsistent,unlabeled,
			    a,lin,&timing_profile,
			    &maxdiff,(long)-1,(long)1);
    if(verbosity >= 1) 
     CIO::message("done.\n");
    
    
    /* after all leave-one-out computed */
    model->loo_error=100.0*loo_count/(double)totdoc;
    model->loo_recall=(1.0-(double)loo_count_pos/(double)trainpos)*100.0;
    model->loo_precision=(trainpos-loo_count_pos)/
      (double)(trainpos-loo_count_pos+loo_count_neg)*100.0;
    if(verbosity >= 1) {
      CIO::message("Leave-one-out estimate of the error: error=%.2f%%\n",
	      model->loo_error);
      CIO::message("Leave-one-out estimate of the recall: recall=%.2f%%\n",
	      model->loo_recall);
      CIO::message("Leave-one-out estimate of the precision: precision=%.2f%%\n",
	      model->loo_precision);
      CIO::message("Actual leave-one-outs computed:  %ld (rho=%.2f)\n",
	      loocomputed,learn_parm->rho);
     CIO::message("Runtime for leave-one-out in cpu-seconds: %.2f\n",
	     (double)(get_runtime()-runtime_start_loo)/100.0);
    }
  }
    
  if(learn_parm->alphafile[0])
    write_alphas(learn_parm->alphafile,a,label,totdoc);
  
  shrink_state_cleanup(&shrink_state);
  free(inconsistent);
  free(unlabeled);
  free(a);
  free(a_fullset);
  free(xi_fullset);
  free(lin);
  free(learn_parm->svm_cost);
}

long CSVMLight::optimize_to_convergence(
DOC *docs,                 /* Training vectors (x-part) */
long *label,               /* Training labels (y-part, zero if test example for transduction) */ 
long totdoc,               /* Number of examples in docs/label */
long totwords,             /* Number of features (i.e. highest feature index) */
LEARN_PARM *learn_parm,    /* Learning paramenters */
KERNEL_PARM *kernel_parm,  /* Kernel paramenters */
KERNEL_CACHE *kernel_cache,/* Initialed Cache */
SHRINK_STATE *shrink_state,/* State of active variables */
MODEL *model,              /* Returns learning result */
long *inconsistent,
long *unlabeled,
double *a,
double *lin,
TIMING *timing_profile,
double *maxdiff,
long heldout,
long retrain)
{
  long *chosen,*key,i,j,jj,*last_suboptimal_at,noshrink;
  long inconsistentnum,choosenum,already_chosen=0,iteration;
  long misclassified,supvecnum=0,*active2dnum,inactivenum;
  long *working2dnum,*selexam;
  long activenum;
  double criterion,eq;
  double *a_old;
  long t0=0,t1=0,t2=0,t3=0,t4=0,t5=0,t6=0; /* timing */
  long transductcycle;
  long transduction;
  double epsilon_crit_org; 

  double *selcrit;  /* buffer for sorting */        
  CFLOAT *aicache;  /* buffer to keep one row of hessian */
////  double *weights;  /* buffer for weight vector in linear case */
  QP qp;            /* buffer for one quadratic program */

  epsilon_crit_org=learn_parm->epsilon_crit; /* save org */
  if(kernel_parm->kernel_type == LINEAR) {
    learn_parm->epsilon_crit=2.0;
    kernel_cache=NULL;   /* caching makes no sense for linear kernel */
  } 
  learn_parm->epsilon_shrink=2;
  (*maxdiff)=1;

////  learn_parm->totwords=totwords;

  chosen = (long *)my_malloc(sizeof(long)*totdoc);
  last_suboptimal_at = (long *)my_malloc(sizeof(long)*totdoc);
  key = (long *)my_malloc(sizeof(long)*(totdoc+11)); 
  selcrit = (double *)my_malloc(sizeof(double)*totdoc);
  selexam = (long *)my_malloc(sizeof(long)*totdoc);
  a_old = (double *)my_malloc(sizeof(double)*totdoc);
  aicache = (CFLOAT *)my_malloc(sizeof(CFLOAT)*totdoc);
  working2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  active2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
  qp.opt_ce = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_ce0 = (double *)my_malloc(sizeof(double));
  qp.opt_g = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize
				 *learn_parm->svm_maxqpsize);
  qp.opt_g0 = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_xinit = (double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_low=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
  qp.opt_up=(double *)my_malloc(sizeof(double)*learn_parm->svm_maxqpsize);
////  weights=(double *)my_malloc(sizeof(double)*(totwords+1));

  choosenum=0;
  inconsistentnum=0;
  transductcycle=0;
  transduction=0;
  if(!retrain) retrain=1;
  iteration=1;

  if(kernel_cache) {
    kernel_cache->time=iteration;  /* for lru cache */
    kernel_cache_reset_lru(kernel_cache);
  }

  for(i=0;i<totdoc;i++) {    /* various inits */
    chosen[i]=0;
    a_old[i]=a[i];
    last_suboptimal_at[i]=1;
    if(inconsistent[i]) 
      inconsistentnum++;
    if(unlabeled[i]) {
      transduction=1;
    }
  }
  activenum=compute_index(shrink_state->active,totdoc,active2dnum);
  inactivenum=totdoc-activenum;
  clear_index(working2dnum);

                            /* repeat this loop until we have convergence */
  for(;retrain && iteration < 1000000 ;iteration++) {

    if(kernel_cache)
      kernel_cache->time=iteration;  /* for lru cache */
    if(verbosity>=2) {
     CIO::message(
	"Iteration %ld: ",iteration);
    }
    else if(verbosity==1) {
     CIO::message(".");
    }

    if(verbosity>=2) t0=get_runtime();
    if(verbosity>=3) {
     CIO::message("\nSelecting working set... "); 
    }

    i=0;
    for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* clear working set */
      if((chosen[j]>=(learn_parm->svm_maxqpsize/
		      math.min(learn_parm->svm_maxqpsize,
			   learn_parm->svm_newvarsinqp))) 
	 || (inconsistent[j])
	 || (j == heldout)) {
	chosen[j]=0; 
	choosenum--; 
      }
      else {
	chosen[j]++;
	working2dnum[i++]=j;
      }
    }
    working2dnum[i]=-1;

    if(retrain == 2) {
      choosenum=0;
      for(jj=0;(j=working2dnum[jj])>=0;jj++) { /* fully clear working set */
	chosen[j]=0; 
      }
      clear_index(working2dnum);
      for(i=0;i<totdoc;i++) { /* set inconsistent examples to zero (-i 1) */
	if((inconsistent[i] || (heldout==i)) && (a[i] != 0.0)) {
	  chosen[i]=99999;
	  choosenum++;
	  a[i]=0;
	}
      }
      if(learn_parm->biased_hyperplane) {
	eq=0;
	for(i=0;i<totdoc;i++) { /* make sure we fulfill equality constraint */
	  eq+=a[i]*label[i];
	}
	for(i=0;(i<totdoc) && (fabs(eq) > learn_parm->epsilon_a);i++) {
	  if((eq*label[i] > 0) && (a[i] > 0)) {
	    chosen[i]=88888;
	    choosenum++;
	    if((eq*label[i]) > a[i]) {
	      eq-=(a[i]*label[i]);
	      a[i]=0;
	    }
	    else {
	      a[i]-=(eq*label[i]);
	      eq=0;
	    }
	  }
	}
      }
      compute_index(chosen,totdoc,working2dnum);
    }
    else {      /* select working set according to steepest gradient */ 
      if((math.min(learn_parm->svm_newvarsinqp,learn_parm->svm_maxqpsize)>=4) 
	 && (kernel_parm->kernel_type != LINEAR)) {
	/* select part of the working set from cache */
	already_chosen=select_next_qp_subproblem_grad_cache(
                              label,unlabeled,a,lin,totdoc,
                              math.min((long)(learn_parm->svm_maxqpsize-choosenum),
				   (long)(learn_parm->svm_newvarsinqp/2)),
			      learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,
			      key,chosen);
	choosenum+=already_chosen;
      }
      choosenum+=select_next_qp_subproblem_grad(label,unlabeled,a,lin,totdoc,
                              math.min((long)(learn_parm->svm_maxqpsize-choosenum),
				   (long)(learn_parm->svm_newvarsinqp-already_chosen)),
                              learn_parm,inconsistent,active2dnum,
			      working2dnum,selcrit,selexam,kernel_cache,key,
			      chosen);
    }

    if(verbosity>=2) {
     CIO::message(" %ld vectors chosen\n",choosenum); 
    }

    if(verbosity>=2) t1=get_runtime();

    if(kernel_cache) 
      cache_multiple_kernel_rows(kernel_cache,docs,working2dnum,
				 choosenum,kernel_parm); 
    
    if(verbosity>=2) t2=get_runtime();
    if(retrain != 2) {
      optimize_svm(docs,label,unlabeled,chosen,active2dnum,model,totdoc,
		   working2dnum,choosenum,a,lin,learn_parm,aicache,
		   kernel_parm,&qp,&epsilon_crit_org);
    }

    if(verbosity>=2) t3=get_runtime();
    update_linear_component(docs,label,active2dnum,a,a_old,working2dnum,totdoc,
			    totwords,kernel_parm,kernel_cache,lin,aicache,
				NULL);
////			    weights);

    if(verbosity>=2) t4=get_runtime();
    supvecnum=calculate_svm_model(docs,label,unlabeled,lin,a,a_old,learn_parm,
				  working2dnum,model);

    if(verbosity>=2) t5=get_runtime();

    /* The following computation of the objective function works only */
    /* relative to the active variables */
    if(verbosity>=3) {
      criterion=compute_objective_function(a,lin,label,active2dnum);
     CIO::message("Objective function (over active variables): %.16f\n",criterion);
      
    }

    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      a_old[i]=a[i];
    }

    if(retrain == 2) {  /* reset inconsistent unlabeled examples */
      for(i=0;(i<totdoc);i++) {
	if(inconsistent[i] && unlabeled[i]) {
	  inconsistent[i]=0;
	  label[i]=0;
	}
      }
    }

    retrain=check_optimality(model,label,unlabeled,a,lin,totdoc,learn_parm,
			     maxdiff,epsilon_crit_org,&misclassified,
			     inconsistent,active2dnum,last_suboptimal_at,
			     iteration,kernel_parm);

    if(verbosity>=2) {
      t6=get_runtime();
      timing_profile->time_select+=t1-t0;
      timing_profile->time_kernel+=t2-t1;
      timing_profile->time_opti+=t3-t2;
      timing_profile->time_update+=t4-t3;
      timing_profile->time_model+=t5-t4;
      timing_profile->time_check+=t6-t5;
    }

    noshrink=0;
    if((!retrain) && (inactivenum>0) 
       && ((!learn_parm->skip_final_opt_check) 
	   || (kernel_parm->kernel_type == LINEAR))) { 
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	if(verbosity==1) {
	 CIO::message("\n");
	}
	printf(" Checking optimality of inactive variables..."); 
	
      }
      t1=get_runtime();
      reactivate_inactive_examples(label,unlabeled,a,shrink_state,lin,totdoc,
				   totwords,iteration,learn_parm,inconsistent,
				   docs,kernel_parm,kernel_cache,model,aicache,
				   NULL,maxdiff);
////				   weights,maxdiff);
      /* Update to new active variables. */
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=totdoc-activenum;
      /* termination criterion */
      noshrink=1;
      retrain=0;
      if((*maxdiff) > learn_parm->epsilon_crit) 
	retrain=1;
      timing_profile->time_shrink+=get_runtime()-t1;
      if(((verbosity>=1) && (kernel_parm->kernel_type != LINEAR)) 
	 || (verbosity>=2)) {
	printf("done.\n"); 
       CIO::message(" Number of inactive variables = %ld\n",inactivenum);
      }		  
    }

    if((!retrain) && (learn_parm->epsilon_crit>(*maxdiff))) 
      learn_parm->epsilon_crit=(*maxdiff);
    if((!retrain) && (learn_parm->epsilon_crit>epsilon_crit_org)) {
      learn_parm->epsilon_crit/=2.0;
      retrain=1;
      noshrink=1;
    }
    if(learn_parm->epsilon_crit<epsilon_crit_org) 
      learn_parm->epsilon_crit=epsilon_crit_org;
    
    if(verbosity>=2) {
     CIO::message(" => (%ld SV (incl. %ld SV at u-bound), max violation=%.5f)\n",
	     supvecnum,model->at_upper_bound,(*maxdiff)); 
     
    }
    if(verbosity>=3) {
     CIO::message("\n");
    }

    if((!retrain) && (transduction)) {
      for(i=0;(i<totdoc);i++) {
	shrink_state->active[i]=1;
      }
      activenum=compute_index(shrink_state->active,totdoc,active2dnum);
      inactivenum=0;
      if(verbosity==1)CIO::message("done\n");
      retrain=incorporate_unlabeled_examples(model,label,inconsistent,
					     unlabeled,a,lin,totdoc,
					     selcrit,selexam,key,
					     transductcycle,kernel_parm,
					     learn_parm);
      epsilon_crit_org=learn_parm->epsilon_crit;
      if(kernel_parm->kernel_type == LINEAR)
	learn_parm->epsilon_crit=1; 
      transductcycle++;
    } 
    else if(((iteration % 10) == 0) && (!noshrink)) {
      activenum=shrink_problem(learn_parm,shrink_state,active2dnum,last_suboptimal_at,
			       iteration,totdoc,math.max((long)(activenum/10),(long) 100),
			       a,inconsistent);
      inactivenum=totdoc-activenum;
      if((kernel_cache)
	 && (supvecnum>kernel_cache->max_elems)
	 && ((kernel_cache->activenum-activenum)>math.max((long)(activenum/10),(long) 500))) {
	kernel_cache_shrink(kernel_cache,totdoc,math.max((long)(activenum/10),(long) 500),
			    shrink_state->active); 
      }
    }

    if((!retrain) && learn_parm->remove_inconsistent) {
      if(verbosity>=1) {
	printf(" Moving training errors to inconsistent examples...");
	
      }
      if(learn_parm->remove_inconsistent == 1) {
	retrain=identify_inconsistent(a,label,unlabeled,totdoc,learn_parm,
				      &inconsistentnum,inconsistent); 
      }
      else if(learn_parm->remove_inconsistent == 2) {
	retrain=identify_misclassified(lin,label,unlabeled,totdoc,
				       model,&inconsistentnum,inconsistent); 
      }
      else if(learn_parm->remove_inconsistent == 3) {
	retrain=identify_one_misclassified(lin,label,unlabeled,totdoc,
				   model,&inconsistentnum,inconsistent);
      }
      if(retrain) {
	if(kernel_parm->kernel_type == LINEAR) { /* reinit shrinking */
	  learn_parm->epsilon_crit=2.0;
	} 
      }
      if(verbosity>=1) {
	printf("done.\n");
	if(retrain) {
	 CIO::message(" Now %ld inconsistent examples.\n",inconsistentnum);
	}
      }
    }
  } /* end of loop */

  free(chosen);
  free(last_suboptimal_at);
  free(key);
  free(selcrit);
  free(selexam);
  free(a_old);
  free(aicache);
  free(working2dnum);
  free(active2dnum);
  free(qp.opt_ce);
  free(qp.opt_ce0);
  free(qp.opt_g);
  free(qp.opt_g0);
  free(qp.opt_xinit);
  free(qp.opt_low);
  free(qp.opt_up);
////  free(weights);

  learn_parm->epsilon_crit=epsilon_crit_org; /* restore org */

  return(iteration);
}

/* Return value of objective function. */
/* Works only relative to the active variables! */
double CSVMLight::compute_objective_function(double *a, double* lin, long *label, long* active2dnum)
{
  long i,ii;
  double criterion;
  /* calculate value of objective function */
  criterion=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    criterion=criterion-a[i]+0.5*a[i]*label[i]*lin[i];
  } 
  return(criterion);
}

/* initializes and empties index */
void CSVMLight::clear_index(long *index)  
{
  index[0]=-1;
} 

/* initializes and empties index */
void CSVMLight::add_to_index(long *index, long elem)
{
  register long i;
  for(i=0;index[i] != -1;i++);
  index[i]=elem;
  index[i+1]=-1;
}

/* create an inverted index of binfeature */
long CSVMLight::compute_index(long *binfeature, long range, long *index)
{               
  register long i,ii;

  ii=0;
  for(i=0;i<range;i++) {
    if(binfeature[i]) {
      index[ii]=i;
      ii++;
    }
  }
  for(i=0;i<4;i++) {
    index[ii+i]=-1;
  }
  return(ii);
}

void CSVMLight::optimize_svm(
DOC *docs,               /* Do optimization on the working set. */
long *label, long *unlabeled, long *chosen, long *active2dnum,
MODEL *model,
long totdoc, long* working2dnum, long varnum,
double *a, double* lin,
LEARN_PARM *learn_parm,
CFLOAT *aicache,
KERNEL_PARM *kernel_parm,
QP *qp,
double *epsilon_crit_target)
{
    long i;
    double *a_v;

    compute_matrices_for_optimization(docs,label,unlabeled,chosen,active2dnum,
				      working2dnum,model,a,lin,varnum,
				      totdoc,learn_parm,aicache,kernel_parm,
				      qp);

    if(verbosity>=3) {
     CIO::message("Running optimizer...");
    }
    /* call the qp-subsolver */
    a_v=optimize_qp(qp,epsilon_crit_target,
		    learn_parm->svm_maxqpsize,
		    &(model->b),   /* in case the optimizer gives us */
                                   /* the threshold for free. otherwise */
                                   /* b is calculated in calculate_model. */
		    learn_parm);
    if(verbosity>=3) {         
     CIO::message("done\n");
    }

    for(i=0;i<varnum;i++) {
      a[working2dnum[i]]=a_v[i];
      /*
      if(a_v[i]<=(0+learn_parm->epsilon_a)) {
	a[working2dnum[i]]=0;
      }
      else if(a_v[i]>=(learn_parm->svm_cost[working2dnum[i]]-learn_parm->epsilon_a)) {
	a[working2dnum[i]]=learn_parm->svm_cost[working2dnum[i]];
      }
      */
    }
}

void CSVMLight::compute_matrices_for_optimization(
DOC *docs,
long *label, long* unlabeled,long* chosen,long* active2dnum, long* key,
MODEL *model,
double *a, double* lin,
long varnum, long totdoc,
LEARN_PARM *learn_parm,
CFLOAT *aicache,
KERNEL_PARM *kernel_parm,
QP *qp)
{
  register long ki,kj,i,j;
  register double kernel_temp;

  if(verbosity>=3) {
    CIO::message("Computing qp-matrices (type %ld kernel [degree %ld, rbf_gamma %f, coef_lin %f, coef_const %f])...",kernel_parm->kernel_type,kernel_parm->poly_degree,kernel_parm->rbf_gamma,kernel_parm->coef_lin,kernel_parm->coef_const); 
   
  }

  qp->opt_n=varnum;
  qp->opt_ce0[0]=0; /* compute the constant for equality constraint */
  for(j=1;j<model->sv_num;j++) { /* start at 1 */
    if(!chosen[(model->supvec[j])->docnum]) {
      qp->opt_ce0[0]+=model->alpha[j];
    }
  } 
  if(learn_parm->biased_hyperplane) 
    qp->opt_m=1;
  else 
    qp->opt_m=0;  /* eq-constraint will be ignored */

  /* init linear part of objective function */
  for(i=0;i<varnum;i++) {
    qp->opt_g0[i]=lin[key[i]];
  }

  for(i=0;i<varnum;i++) {
    ki=key[i];

    /* Compute the matrix for equality constraints */
    qp->opt_ce[i]=label[ki];
    qp->opt_low[i]=0;
    qp->opt_up[i]=learn_parm->svm_cost[ki];

    kernel_temp=(double)kernel(kernel_parm,&(docs[ki]),&(docs[ki])); 
    /* compute linear part of objective function */
    qp->opt_g0[i]-=(kernel_temp*a[ki]*(double)label[ki]); 
    /* compute quadratic part of objective function */
    qp->opt_g[varnum*i+i]=kernel_temp;
    for(j=i+1;j<varnum;j++) {
      kj=key[j];
      kernel_temp=(double)kernel(kernel_parm,&(docs[ki]),&(docs[kj]));
      /* compute linear part of objective function */
      qp->opt_g0[i]-=(kernel_temp*a[kj]*(double)label[kj]);
      qp->opt_g0[j]-=(kernel_temp*a[ki]*(double)label[ki]); 
      /* compute quadratic part of objective function */
      qp->opt_g[varnum*i+j]=(double)label[ki]*(double)label[kj]*kernel_temp;
      qp->opt_g[varnum*j+i]=(double)label[ki]*(double)label[kj]*kernel_temp;
    }

    if(verbosity>=3) {
      if(i % 20 == 0) {
	CIO::message("%ld..",i);
      }
    }
  }

  for(i=0;i<varnum;i++) {
    /* assure starting at feasible point */
    qp->opt_xinit[i]=a[key[i]];
    /* set linear part of objective function */
    qp->opt_g0[i]=-1.0+qp->opt_g0[i]*(double)label[key[i]];    
  }

  if(verbosity>=3) {
    CIO::message("done\n");
  }
}

long CSVMLight::calculate_svm_model(
DOC *docs,              /* Compute decision function based on current values */
long *label, long* unlabeled, /* of alpha. */
double *lin, double *a, double* a_old,
LEARN_PARM *learn_parm,
long *working2dnum,
MODEL *model)
{
  long i,ii,pos,b_calculated=0;
  double ex_c;

  if(verbosity>=3) {
   CIO::message("Calculating model...");
  }

  if(!learn_parm->biased_hyperplane) {
    model->b=0;
    b_calculated=1;
  }

  for(ii=0;(i=working2dnum[ii])>=0;ii++) {
    if((a_old[i]>0) && (a[i]==0)) { /* remove from model */
      pos=model->index[i]; 
      model->index[i]=-1;
      (model->sv_num)--;
      model->supvec[pos]=model->supvec[model->sv_num];
      model->alpha[pos]=model->alpha[model->sv_num];
      model->index[(model->supvec[pos])->docnum]=pos;
    }
    else if((a_old[i]==0) && (a[i]>0)) { /* add to model */
      model->supvec[model->sv_num]=&(docs[i]);
      model->alpha[model->sv_num]=a[i]*(double)label[i];
      model->index[i]=model->sv_num;
      (model->sv_num)++;
    }
    else if(a_old[i]==a[i]) { /* nothing to do */
    }
    else {  /* just update alpha */
      model->alpha[model->index[i]]=a[i]*(double)label[i];
    }
      
    ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
    if((a_old[i]>=ex_c) && (a[i]<ex_c)) { 
      (model->at_upper_bound)--;
    }
    else if((a_old[i]<ex_c) && (a[i]>=ex_c)) { 
      (model->at_upper_bound)++;
    }

    if((!b_calculated) 
       && (a[i]>learn_parm->epsilon_a) && (a[i]<ex_c)) {   /* calculate b */
	model->b=(-(double)label[i]+lin[i]);
	b_calculated=1;
    }
  }      

  if(verbosity>=3) {
   CIO::message("done\n");
  }

  /* If there is no alpha in the working set not at bounds, then
     just use the model->b from the last iteration or the one provided 
     by the core optimizer */

  return(model->sv_num-1); /* have to substract one, since element 0 is empty*/
}

/* Check KT-conditions */
long CSVMLight::check_optimality(
MODEL *model,            
long *label, long* unlabeled,
double *a, double* lin,
long totdoc,
LEARN_PARM *learn_parm,
double *maxdiff, double epsilon_crit_org,
long *misclassified,
long *inconsistent,long* active2dnum, long *last_suboptimal_at, long iteration,
KERNEL_PARM *kernel_parm)
{
  long i,ii,retrain;
  double dist,ex_c;

  if(kernel_parm->kernel_type == LINEAR) {  /* be optimistic */
    learn_parm->epsilon_shrink=-learn_parm->epsilon_crit+epsilon_crit_org;  
  }
  else {  /* be conservative */
    learn_parm->epsilon_shrink=learn_parm->epsilon_shrink*0.7+(*maxdiff)*0.3; 
  }
  retrain=0;
  (*maxdiff)=0;
  (*misclassified)=0;
  for(ii=0;(i=active2dnum[ii])>=0;ii++) {
    if((!inconsistent[i]) && label[i]) {
      dist=(lin[i]-model->b)*(double)label[i];/* 'distance' from hyperplane*/
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if(dist <= 0) {       
	(*misclassified)++;  /* does not work due to deactivation of var */
      }
      if((a[i]>learn_parm->epsilon_a) && (dist > 1)) {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      /* Count how long a variable was at lower/upper bound (and optimal).*/
      /* Variables, which were at the bound and optimal for a long */
      /* time are unlikely to become support vectors. In case our */
      /* cache is filled up, those variables are excluded to save */
      /* kernel evaluations. (See chapter 'Shrinking').*/ 
      if((a[i]>(learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	last_suboptimal_at[i]=iteration;         /* not at bound */
      }
      else if((a[i]<=(learn_parm->epsilon_a)) 
	      && (dist < (1.0+learn_parm->epsilon_shrink))) {
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
      else if((a[i]>=ex_c)
	      && (dist > (1.0-learn_parm->epsilon_shrink)))  { 
	last_suboptimal_at[i]=iteration;         /* not likely optimal */
      }
    }   
  }
  /* termination criterion */
  if((!retrain) && ((*maxdiff) > learn_parm->epsilon_crit)) {  
    retrain=1;
  }
  return(retrain);
}

long CSVMLight::identify_inconsistent(
double *a,
long *label,long* unlabeled, long totdoc,
LEARN_PARM *learn_parm,
long *inconsistentnum, long* inconsistent)
{
  long i,retrain;

  /* Throw out examples with multipliers at upper bound. This */
  /* corresponds to the -i 1 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    if((!inconsistent[i]) && (!unlabeled[i]) 
       && (a[i]>=(learn_parm->svm_cost[i]-learn_parm->epsilon_a))) { 
	(*inconsistentnum)++;
	inconsistent[i]=1;  /* never choose again */
	retrain=2;          /* start over */
	if(verbosity>=3) {
	 CIO::message("inconsistent(%ld)..",i);
	}
    }
  }
  return(retrain);
}

long CSVMLight::identify_misclassified(
double *lin,
long *label, long* unlabeled, long totdoc,
MODEL *model,
long *inconsistentnum, long* inconsistent)
{
  long i,retrain;
  double dist;

  /* Throw out misclassified examples. This */
  /* corresponds to the -i 2 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    dist=(lin[i]-model->b)*(double)label[i]; /* 'distance' from hyperplane*/  
    if((!inconsistent[i]) && (!unlabeled[i]) && (dist <= 0)) { 
	(*inconsistentnum)++;
	inconsistent[i]=1;  /* never choose again */
	retrain=2;          /* start over */
	if(verbosity>=3) {
	 CIO::message("inconsistent(%ld)..",i);
	}
    }
  }
  return(retrain);
}

long CSVMLight::identify_one_misclassified(
double *lin,
long *label, long* unlabeled, long totdoc,
MODEL *model,
long *inconsistentnum, long *inconsistent)
{
  long i,retrain,maxex=-1;
  double dist,maxdist=0;

  /* Throw out the 'most misclassified' example. This */
  /* corresponds to the -i 3 option. */
  /* ATTENTION: this is just a heuristic for finding a close */
  /*            to minimum number of examples to exclude to */
  /*            make the problem separable with desired margin */
  retrain=0;
  for(i=0;i<totdoc;i++) {
    if((!inconsistent[i]) && (!unlabeled[i])) {
      dist=(lin[i]-model->b)*(double)label[i];/* 'distance' from hyperplane*/  
      if(dist<maxdist) {
	maxdist=dist;
	maxex=i;
      }
    }
  }
  if(maxex>=0) {
    (*inconsistentnum)++;
    inconsistent[maxex]=1;  /* never choose again */
    retrain=2;          /* start over */
    if(verbosity>=3) {
     CIO::message("inconsistent(%ld)..",i);
    }
  }
  return(retrain);
}

void CSVMLight::update_linear_component(
DOC *docs,
long *label, 
long *active2dnum,                  /* keep track of the linear component */
double *a, double* a_old,                   /* lin of the gradient etc. by updating */
long *working2dnum, long totdoc, long totwords, /* based on the change of the variables */
KERNEL_PARM *kernel_parm,           /* in the current working set */
KERNEL_CACHE *kernel_cache,
double *lin,
CFLOAT *aicache,
double *weights)
{
  register long i,ii,j,jj;
  register double tec;

////  if(kernel_parm->kernel_type==0) { /* special linear case */
////    clear_vector_n(weights,totwords);
////    for(ii=0;(i=working2dnum[ii])>=0;ii++) {
////      if(a[i] != a_old[i]) {
////	add_vector_ns(weights,docs[i].words,((a[i]-a_old[i])*(double)label[i]));
////      }
////    }
////    for(jj=0;(j=active2dnum[jj])>=0;jj++) {
////      lin[j]+=sprod_ns(weights,docs[j].words);
////    }
////  }
////  else {                            /* general case */
    for(jj=0;(i=working2dnum[jj])>=0;jj++) {
      if(a[i] != a_old[i]) {
	get_kernel_row(kernel_cache,docs,i,totdoc,active2dnum,aicache,
		       kernel_parm);
	for(ii=0;(j=active2dnum[ii])>=0;ii++) {
	  tec=aicache[j];
	  lin[j]+=(((a[i]*tec)-(a_old[i]*tec))*(double)label[i]);
	}
    }
    }
////  }
}


long CSVMLight::incorporate_unlabeled_examples(
MODEL *model,
long *label,
long *inconsistent, long* unlabeled,
double *a, double* lin,
long totdoc,
double *selcrit,
long *select, long* key,long transductcycle,
KERNEL_PARM *kernel_parm,
LEARN_PARM *learn_parm)
{
  long i,j,k,j1,j2,j3,j4,unsupaddnum1=0,unsupaddnum2=0;
  long pos,neg,upos,uneg,orgpos,orgneg,nolabel,newpos,newneg,allunlab;
  double dist,model_length,posratio,negratio;
  long check_every=2;
  double loss;
  static double switchsens=0.0,switchsensorg=0.0;
  double umin,umax,sumalpha;
  long imin=0,imax=0;
  static long switchnum=0;

  switchsens/=1.2;

  /* assumes that lin[] is up to date -> no inactive vars */

  orgpos=0;
  orgneg=0;
  newpos=0;
  newneg=0;
  nolabel=0;
  allunlab=0;
  for(i=0;i<totdoc;i++) {
    if(!unlabeled[i]) {
      if(label[i] > 0) {
	orgpos++;
      }
      else {
	orgneg++;
      }
    }
    else {
      allunlab++;
      if(unlabeled[i]) {
	if(label[i] > 0) {
	  newpos++;
	}
	else if(label[i] < 0) {
	  newneg++;
	}
      }
    }
    if(label[i]==0) {
      nolabel++;
    }
  }

  if(learn_parm->transduction_posratio >= 0) {
    posratio=learn_parm->transduction_posratio;
  }
  else {
    posratio=(double)orgpos/(double)(orgpos+orgneg); /* use ratio of pos/neg */
  }                                                  /* in training data */
  negratio=1.0-posratio;

  learn_parm->svm_costratio=1.0;                     /* global */
  if(posratio>0) {
    learn_parm->svm_costratio_unlab=negratio/posratio;
  }
  else {
    learn_parm->svm_costratio_unlab=1.0;
  }
  
  pos=0;
  neg=0;
  upos=0;
  uneg=0;
  for(i=0;i<totdoc;i++) {
    dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
    if(dist>0) {
      pos++;
    }
    else {
      neg++;
    }
    if(unlabeled[i]) {
      if(dist>0) {
	upos++;
      }
      else {
	uneg++;
      }
    }
    if((!unlabeled[i]) && (a[i]>(learn_parm->svm_cost[i]-learn_parm->epsilon_a))) {
      /*     CIO::message("Ubounded %ld (class %ld, unlabeled %ld)\n",i,label[i],unlabeled[i]); */
    }
  }
  if(verbosity>=2) {
   CIO::message("POS=%ld, ORGPOS=%ld, ORGNEG=%ld\n",pos,orgpos,orgneg);
   CIO::message("POS=%ld, NEWPOS=%ld, NEWNEG=%ld\n",pos,newpos,newneg);
   CIO::message("pos ratio = %f (%f).\n",(double)(upos)/(double)(allunlab),posratio);
   
  }

  if(transductcycle == 0) {
    j1=0; 
    j2=0;
    j4=0;
    for(i=0;i<totdoc;i++) {
      dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
      if((label[i]==0) && (unlabeled[i])) {
	selcrit[j4]=dist;
	key[j4]=i;
	j4++;
      }
    }
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    select_top_n(selcrit,j4,select,(long)(allunlab*posratio+0.5));
    for(k=0;(k<(long)(allunlab*posratio+0.5));k++) {
      i=key[select[k]];
      label[i]=1;
      unsupaddnum1++;	
      j1++;
    }
    for(i=0;i<totdoc;i++) {
      if((label[i]==0) && (unlabeled[i])) {
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }
    if(verbosity>=1) {
      /*CIO::message("costratio %lf, costratio_unlab %lf, unlabbound %lf\n",
	 learn_parm->svm_costratio,learn_parm->svm_costratio_unlab,
	 learn_parm->svm_unlabbound); */
     CIO::message("Classifying unlabeled data as %ld POS / %ld NEG.\n",
	     unsupaddnum1,unsupaddnum2); 
     
    }
    if(verbosity >= 1) 
     CIO::message("Retraining.");
    if(verbosity >= 2)CIO::message("\n");
    return((long)3);
  }
  if((transductcycle % check_every) == 0) {
    if(verbosity >= 1) 
     CIO::message("Retraining.");
    if(verbosity >= 2)CIO::message("\n");
    j1=0;
    j2=0;
    unsupaddnum1=0;
    unsupaddnum2=0;
    for(i=0;i<totdoc;i++) {
      if((unlabeled[i] == 2)) {
	unlabeled[i]=1;
	label[i]=1;
	j1++;
	unsupaddnum1++;
      }
      else if((unlabeled[i] == 3)) {
	unlabeled[i]=1;
	label[i]=-1;
	j2++;
	unsupaddnum2++;
      }
    }
    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }

    if(verbosity>=2) {
      /*CIO::message("costratio %lf, costratio_unlab %lf, unlabbound %lf\n",
	     learn_parm->svm_costratio,learn_parm->svm_costratio_unlab,
	     learn_parm->svm_unlabbound); */
     CIO::message("%ld positive -> Added %ld POS / %ld NEG unlabeled examples.\n",
	     upos,unsupaddnum1,unsupaddnum2); 
     
    }

    if(learn_parm->svm_unlabbound == 1) {
      learn_parm->epsilon_crit=0.001; /* do the last run right */
    }
    else {
      learn_parm->epsilon_crit=0.01; /* otherwise, no need to be so picky */
    }

    return((long)3);
  }
  else if(((transductcycle % check_every) < check_every)) { 
    model_length=0;
    sumalpha=0;
    loss=0;
    for(i=0;i<totdoc;i++) {
      model_length+=a[i]*label[i]*lin[i];
      sumalpha+=a[i];
      dist=(lin[i]-model->b);  /* 'distance' from hyperplane*/
      if((label[i]*dist)<(1.0-learn_parm->epsilon_crit)) {
	loss+=(1.0-(label[i]*dist))*learn_parm->svm_cost[i]; 
      }
    }
    model_length=sqrt(model_length); 
    if(verbosity>=2) {
     CIO::message("Model-length = %f (%f), loss = %f, objective = %f\n",
	     model_length,sumalpha,loss,loss+0.5*model_length*model_length);
     
    }
    j1=0;
    j2=0;
    j3=0;
    j4=0;
    unsupaddnum1=0;	
    unsupaddnum2=0;	
    umin=99999;
    umax=-99999;
    j4=1;
    while(j4) {
      umin=99999;
      umax=-99999;
      for(i=0;(i<totdoc);i++) { 
	dist=(lin[i]-model->b);  
	if((label[i]>0) && (unlabeled[i]) && (!inconsistent[i]) 
	   && (dist<umin)) {
	  umin=dist;
	  imin=i;
	}
	if((label[i]<0) && (unlabeled[i])  && (!inconsistent[i]) 
	   && (dist>umax)) {
	  umax=dist;
	  imax=i;
	}
      }
      if((umin < (umax+switchsens-1E-4))) {
	j1++;
	j2++;
	unsupaddnum1++;	
	unlabeled[imin]=3;
	inconsistent[imin]=1;
	unsupaddnum2++;	
	unlabeled[imax]=2;
	inconsistent[imax]=1;
      }
      else
	j4=0;
      j4=0;
    }
    for(j=0;(j<totdoc);j++) {
      if(unlabeled[j] && (!inconsistent[j])) {
	if(label[j]>0) {
	  unlabeled[j]=2;
	}
	else if(label[j]<0) {
	  unlabeled[j]=3;
	}
	/* inconsistent[j]=1; */
	j3++;
      }
    }
    switchnum+=unsupaddnum1+unsupaddnum2;

    /* stop and print out current margin
      CIO::message("switchnum %ld %ld\n",switchnum,kernel_parm->poly_degree);
       if(switchnum == 2*kernel_parm->poly_degree) {
       learn_parm->svm_unlabbound=1;
       }
       */

    if((!unsupaddnum1) && (!unsupaddnum2)) {
      if((learn_parm->svm_unlabbound>=1) && ((newpos+newneg) == allunlab)) {
	for(j=0;(j<totdoc);j++) {
	  inconsistent[j]=0;
	  if(unlabeled[j]) unlabeled[j]=1;
	}
	write_prediction(learn_parm->predfile,model,lin,a,unlabeled,label,
			 totdoc,learn_parm);  
	if(verbosity>=1)
	 CIO::message("Number of switches: %ld\n",switchnum);
	return((long)0);
      }
      switchsens=switchsensorg;
      learn_parm->svm_unlabbound*=1.5;
      if(learn_parm->svm_unlabbound>1) {
	learn_parm->svm_unlabbound=1;
      }
      model->at_upper_bound=0; /* since upper bound increased */
      if(verbosity>=1) 
	printf("Increasing influence of unlabeled examples to %f%% .",
	       learn_parm->svm_unlabbound*100.0);
    }
    else if(verbosity>=1) {
     CIO::message("%ld positive -> Switching labels of %ld POS / %ld NEG unlabeled examples.",
	     upos,unsupaddnum1,unsupaddnum2); 
     
    }

    if(verbosity >= 2)CIO::message("\n");
    
    learn_parm->epsilon_crit=0.5; /* don't need to be so picky */

    for(i=0;i<totdoc;i++) {  /* set upper bounds on vars */
      if(unlabeled[i]) {
	if(label[i] == 1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_costratio_unlab*learn_parm->svm_unlabbound;
	}
	else if(label[i] == -1) {
	  learn_parm->svm_cost[i]=learn_parm->svm_c*
	    learn_parm->svm_unlabbound;
	}
      }
    }

    return((long)2);
  }

  return((long)0); 
}

/*************************** Working set selection ***************************/

long CSVMLight::select_next_qp_subproblem_grad(
long *label,long* unlabeled,
double *a,double* lin,      /* Use the feasible direction approach to select the */
long totdoc, long qp_size, /* next qp-subproblem  (see section 'Selecting a good */
LEARN_PARM *learn_parm, /* working set') */
long *inconsistent, long* active2dnum, long* working2dnum,
double *selcrit,
long *select,
KERNEL_CACHE *kernel_cache,
long *key, long *chosen)
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=lin[j]-(double)label[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!inconsistent[j]) 
       && (label[j])
       && (!chosen[j])) {
      selcrit[activedoc]=(double)(label[j])-lin[j];
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

long CSVMLight::select_next_qp_subproblem_grad_cache(
long *label, long *unlabeled,
double *a, double *lin,         /* Use the feasible direction approach to select the */
long totdoc, long qp_size,    /* next qp-subproblem  (see chapter 'Selecting a  */
LEARN_PARM *learn_parm, /* good working set') among the variable with */
long *inconsistent, long* active2dnum, long* working2dnum, /* cached kernel */
double *selcrit,
long *select,
KERNEL_CACHE *kernel_cache,
long *key, long* chosen)
{
  long choosenum,i,j,k,activedoc,inum;
  double s;

  for(inum=0;working2dnum[inum]>=0;inum++); /* find end of index */
  choosenum=0;
  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=-label[j];
    if((kernel_cache->index[j]>=0)
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0)))
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j]))
      {
      selcrit[activedoc]=(double)label[j]*(-1.0+(double)label[j]*lin[j]);
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<(qp_size/2)) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  }

  activedoc=0;
  for(i=0;(j=active2dnum[i])>=0;i++) {
    s=label[j];
    if((kernel_cache->index[j]>=0)
       && (!((a[j]<=(0+learn_parm->epsilon_a)) && (s<0)))
       && (!((a[j]>=(learn_parm->svm_cost[j]-learn_parm->epsilon_a)) 
	     && (s>0))) 
       && (!chosen[j]) 
       && (label[j])
       && (!inconsistent[j])) 
      {
      selcrit[activedoc]=-(double)(label[j]*(-1.0+(double)label[j]*lin[j]));
      key[activedoc]=j;
      activedoc++;
    }
  }
  select_top_n(selcrit,activedoc,select,(long)(qp_size/2));
  for(k=0;(choosenum<qp_size) && (k<(qp_size/2)) && (k<activedoc);k++) {
    i=key[select[k]];
    chosen[i]=1;
    working2dnum[inum+choosenum]=i;
    choosenum+=1;
    kernel_cache_touch(kernel_cache,i); /* make sure it does not get kicked */
                                        /* out of cache */
  } 
  working2dnum[inum+choosenum]=-1; /* complete index */
  return(choosenum);
}

void CSVMLight::select_top_n(double *selcrit, long range,long* select, long n)
{
  register long i,j;

  for(i=0;(i<n) && (i<range);i++) { /* Initialize with the first n elements */
    for(j=i;j>=0;j--) {
      if((j>0) && (selcrit[select[j-1]]<selcrit[i])){
	select[j]=select[j-1];
      }
      else {
	select[j]=i;
	j=-1;
      }
    }
  }
  for(i=n;i<range;i++) {  
    if(selcrit[i]>selcrit[select[n-1]]) {
      for(j=n-1;j>=0;j--) {
	if((j>0) && (selcrit[select[j-1]]<selcrit[i])) {
	  select[j]=select[j-1];
	}
	else {
	  select[j]=i;
	  j=-1;
	}
      }
    }
  }
}      
      

/******************************** Shrinking  *********************************/

void CSVMLight::init_shrink_state(SHRINK_STATE *shrink_state, long totdoc, long maxhistory)
{
  long i;

  shrink_state->deactnum=0;
  shrink_state->active = (long *)my_malloc(sizeof(long)*totdoc);
  shrink_state->inactive_since = (long *)my_malloc(sizeof(long)*totdoc);
  shrink_state->a_history = (double **)my_malloc(sizeof(double *)*10000);

  for(i=0;i<totdoc;i++) { 
    shrink_state->active[i]=1;
    shrink_state->inactive_since[i]=0;
  }
}

void CSVMLight::shrink_state_cleanup(SHRINK_STATE *shrink_state)
{
  free(shrink_state->active);
  free(shrink_state->inactive_since);
  if(shrink_state->deactnum > 0) 
    free(shrink_state->a_history[shrink_state->deactnum-1]);
  free(shrink_state->a_history);
}

/* shrink some variables away */
/* do the shrinking only if at least minshrink variables can be removed */
long CSVMLight::shrink_problem(LEARN_PARM *learn_parm, SHRINK_STATE *shrink_state, long *active2dnum, 
					long *last_suboptimal_at, long iteration, long totdoc, long minshrink, 
					double *a, long *inconsistent)
{
  long i,ii,change,activenum;
  double *a_old;
  
  activenum=0;
  change=0;
  for(ii=0;active2dnum[ii]>=0;ii++) {
    i=active2dnum[ii];
    activenum++;
    if(((iteration-last_suboptimal_at[i])
	> learn_parm->svm_iter_to_shrink) 
       || (inconsistent[i])) {
      change++;
    }
  }
  if(change>=minshrink) { /* shrink only if sufficiently many candidates */
    /* Shrink problem by removing those variables which are */
    /* optimal at a bound for a minimum number of iterations */
    if(verbosity>=2) {
     CIO::message(" Shrinking...");
    }
    a_old=(double *)my_malloc(sizeof(double)*totdoc);
    shrink_state->a_history[shrink_state->deactnum]=a_old;
    for(i=0;i<totdoc;i++) {
      a_old[i]=a[i];
    }
    change=0;
    for(ii=0;active2dnum[ii]>=0;ii++) {
      i=active2dnum[ii];
      if((((iteration-last_suboptimal_at[i])
	   >learn_parm->svm_iter_to_shrink) 
	  || (inconsistent[i]))) {
	shrink_state->active[i]=0;
	shrink_state->inactive_since[i]=shrink_state->deactnum;
	change++;
      }
    }
    activenum=compute_index(shrink_state->active,totdoc,active2dnum);
    shrink_state->deactnum++;
    if(verbosity>=2) {
     CIO::message("done.\n");
     CIO::message(" Number of inactive variables = %ld\n",totdoc-activenum);
    }
  }
  return(activenum);
} 


void CSVMLight::reactivate_inactive_examples(
long *label,long* unlabeled,              /* Make all variables active again */
double *a, 
SHRINK_STATE *shrink_state,
double *lin,                      /* which had been removed by shrinking. */
long totdoc, long totwords, long iteration,      /* Computes lin for those */
LEARN_PARM *learn_parm,              /* variables from scratch. */
long *inconsistent,
DOC *docs,
KERNEL_PARM *kernel_parm,
KERNEL_CACHE *kernel_cache,
MODEL *model,
CFLOAT *aicache,
double *weights, double* maxdiff)
{
  register long i,j,ii,jj,t,*changed2dnum,*inactive2dnum;
  long *changed,*inactive;
  register double kernel_val,*a_old,dist;
  double ex_c;

  changed=(long *)my_malloc(sizeof(long)*totdoc);
  changed2dnum=(long *)my_malloc(sizeof(long)*(totdoc+11));
  inactive=(long *)my_malloc(sizeof(long)*totdoc);
  inactive2dnum=(long *)my_malloc(sizeof(long)*(totdoc+11));
  for(t=shrink_state->deactnum-1;(t>=0) && shrink_state->a_history[t];t--) {
    if(verbosity>=2) {
     CIO::message("%ld..",t);
    }
    a_old=shrink_state->a_history[t];    
    for(i=0;i<totdoc;i++) {
      inactive[i]=((!shrink_state->active[i]) 
		   && (shrink_state->inactive_since[i] == t));
      changed[i]= (a[i] != a_old[i]);
    }
    compute_index(inactive,totdoc,inactive2dnum);
    compute_index(changed,totdoc,changed2dnum);

////    if(kernel_parm->kernel_type == LINEAR) { /* special linear case */
////      clear_vector_n(weights,totwords);
////      for(ii=0;changed2dnum[ii]>=0;ii++) {
////	i=changed2dnum[ii];
////	add_vector_ns(weights,docs[i].words,
////		      ((a[i]-a_old[i])*(double)label[i]));
////      }
////      for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
////	lin[j]+=sprod_ns(weights,docs[j].words);
////      }
////    }
////    else {
      for(ii=0;(i=changed2dnum[ii])>=0;ii++) {
	get_kernel_row(kernel_cache,docs,i,totdoc,inactive2dnum,aicache,
		       kernel_parm);
	for(jj=0;(j=inactive2dnum[jj])>=0;jj++) {
	  kernel_val=aicache[j];
	  lin[j]+=(((a[i]*kernel_val)-(a_old[i]*kernel_val))*(double)label[i]);
	}
      }
////    }
  }
  (*maxdiff)=0;
  for(i=0;i<totdoc;i++) {
    shrink_state->inactive_since[i]=shrink_state->deactnum-1;
    if(!inconsistent[i]) {
      dist=(lin[i]-model->b)*(double)label[i];
      ex_c=learn_parm->svm_cost[i]-learn_parm->epsilon_a;
      if((a[i]>learn_parm->epsilon_a) && (dist > 1)) {
	if((dist-1.0)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=dist-1.0;
      }
      else if((a[i]<ex_c) && (dist < 1)) {
	if((1.0-dist)>(*maxdiff))  /* largest violation */
	  (*maxdiff)=1.0-dist;
      }
      if((a[i]>(0+learn_parm->epsilon_a)) 
	 && (a[i]<ex_c)) { 
	shrink_state->active[i]=1;                         /* not at bound */
      }
      else if((a[i]<=(0+learn_parm->epsilon_a)) && (dist < (1+learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
      else if((a[i]>=ex_c)
	      && (dist > (1-learn_parm->epsilon_shrink))) {
	shrink_state->active[i]=1;
      }
    }
  }
  for(i=0;i<totdoc;i++) {
    (shrink_state->a_history[shrink_state->deactnum-1])[i]=a[i];
  }
  for(t=shrink_state->deactnum-2;(t>=0) && shrink_state->a_history[t];t--) {
      free(shrink_state->a_history[t]);
      shrink_state->a_history[t]=0;
  }
  free(changed);
  free(changed2dnum);
  free(inactive);
  free(inactive2dnum);
}

/****************************** Cache handling *******************************/

void CSVMLight::get_kernel_row(
KERNEL_CACHE *kernel_cache,
DOC *docs,          /* Get's a row of the matrix of kernel values */
long docnum,long totdoc, /* This matrix has the same form as the Hessian, */ 
long *active2dnum,  /* just that the elements are not multiplied by */
CFLOAT *buffer,     /* y_i * y_j * a_i * a_j */
KERNEL_PARM *kernel_parm) /* Takes the values from the cache if available. */
{
  register long i,j,start;
  DOC *ex;

  ex=&(docs[docnum]);
  if(kernel_cache->index[docnum] != -1) { /* is cached? */
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    start=kernel_cache->activenum*kernel_cache->index[docnum];
    for(i=0;(j=active2dnum[i])>=0;i++) {
      if(kernel_cache->totdoc2active[j] >= 0) {
	buffer[j]=kernel_cache->buffer[start+kernel_cache->totdoc2active[j]];
      }
      else {
	buffer[j]=(CFLOAT)kernel(kernel_parm,ex,&(docs[j]));
      }
    }
  }
  else {
    for(i=0;(j=active2dnum[i])>=0;i++) {
      buffer[j]=(CFLOAT)kernel(kernel_parm,ex,&(docs[j]));
    }
  }
}


/* Fills cache for the row m */
void CSVMLight::cache_kernel_row(KERNEL_CACHE *kernel_cache, DOC *docs, long m, KERNEL_PARM *kernel_parm)
{
  register DOC *ex;
  register long j,k,l;
  register CFLOAT *cache;

  if(!kernel_cache_check(kernel_cache,m)) {  /* not cached yet*/
    cache = kernel_cache_clean_and_malloc(kernel_cache,m);
    if(cache) {
      l=kernel_cache->totdoc2active[m];
      ex=&(docs[m]);
      for(j=0;j<kernel_cache->activenum;j++) {  /* fill cache */
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

/* Fills cache for the rows in key */
void CSVMLight::cache_multiple_kernel_rows(KERNEL_CACHE *kernel_cache, DOC *docs, long *key, long varnum, KERNEL_PARM *kernel_parm)
{
  register long i;

  for(i=0;i<varnum;i++) {  /* fill up kernel cache */
    cache_kernel_row(kernel_cache,docs,key[i],kernel_parm);
  }
}

/* remove numshrink columns in the cache */
/* which correspond to examples marked  */
void CSVMLight::kernel_cache_shrink(KERNEL_CACHE *kernel_cache, long totdoc, long numshrink, long *after)
{                           
  register long i,j,jj,from=0,to=0,scount;     /* 0 in after. */
  long *keep;

  if(verbosity>=2) {
   CIO::message(" Reorganizing cache...");
  }

  keep=(long *)my_malloc(sizeof(long)*totdoc);
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

  free(keep);

  if(verbosity>=2) {
   CIO::message("done.\n");
   CIO::message(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
  }
}

void CSVMLight::kernel_cache_init(KERNEL_CACHE *kernel_cache, long totdoc, long buffsize)
{
  long i;

  kernel_cache->index = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->occu = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->lru = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->invindex = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->active2totdoc = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->totdoc2active = (long *)my_malloc(sizeof(long)*totdoc);
  kernel_cache->buffer = (CFLOAT *)my_malloc(buffsize*1024*1024);

  kernel_cache->buffsize=(long)(buffsize*1024*1024/sizeof(CFLOAT));

  kernel_cache->max_elems=(long)(kernel_cache->buffsize/totdoc);
  if(kernel_cache->max_elems>totdoc) {
    kernel_cache->max_elems=totdoc;
  }

  if(verbosity>=2) {
   CIO::message(" Cache-size in rows = %ld\n",kernel_cache->max_elems);
   CIO::message(" Kernel evals so far: %ld\n",kernel_cache_statistic);    
  }

  kernel_cache->elems=0;   /* initialize cache */
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

void CSVMLight::kernel_cache_reset_lru(KERNEL_CACHE *kernel_cache)
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

void CSVMLight::kernel_cache_cleanup(KERNEL_CACHE *kernel_cache)
{
  free(kernel_cache->index);
  free(kernel_cache->occu);
  free(kernel_cache->lru);
  free(kernel_cache->invindex);
  free(kernel_cache->active2totdoc);
  free(kernel_cache->totdoc2active);
  free(kernel_cache->buffer);
}

long CSVMLight::kernel_cache_malloc(KERNEL_CACHE *kernel_cache)
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

void CSVMLight::kernel_cache_free(KERNEL_CACHE *kernel_cache, long i)
{
  kernel_cache->occu[i]=0;
  kernel_cache->elems--;
}

/* remove least recently used cache */
/* element */
long CSVMLight::kernel_cache_free_lru(KERNEL_CACHE *kernel_cache)  
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

/* Get a free cache entry. In case cache is full, the lru */
/* element is removed. */
CFLOAT* CSVMLight::kernel_cache_clean_and_malloc(KERNEL_CACHE *kernel_cache, long docnum)
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
  kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
  return((CFLOAT *)((long)kernel_cache->buffer
		    +(kernel_cache->activenum*sizeof(CFLOAT)*
		      kernel_cache->index[docnum])));
}

/* Update lru time to avoid removal from cache. */
long CSVMLight::kernel_cache_touch(KERNEL_CACHE *kernel_cache, long docnum)
{
  if(kernel_cache && kernel_cache->index[docnum] != -1) {
    kernel_cache->lru[kernel_cache->index[docnum]]=kernel_cache->time; /* lru */
    return(1);
  }
  return(0);
}
  
/* Is that row cached? */
long CSVMLight::kernel_cache_check(KERNEL_CACHE *kernel_cache, long docnum)
{
  return(kernel_cache->index[docnum] != -1);
}
  
/************************** Compute estimates ******************************/

void CSVMLight::compute_xa_estimates(
MODEL *model,                           /* xa-estimate of error rate, */
long *label,long *unlabeled,long totdoc,          /* recall, and precision      */
DOC *docs,       
double *lin,double* a,
KERNEL_PARM *kernel_parm,
LEARN_PARM *learn_parm,
double *error, double* recall, double* precision)
{
  long i,looerror,looposerror,loonegerror;
  long totex,totposex;
  double xi,r_delta,r_delta_sq,sim=0;
  long *sv2dnum=NULL,*sv=NULL,svnum;

  r_delta=estimate_r_delta(docs,totdoc,kernel_parm); 
  r_delta_sq=r_delta*r_delta;

  looerror=0;
  looposerror=0;
  loonegerror=0;
  totex=0;
  totposex=0;
  svnum=0;

  if(learn_parm->xa_depth > 0) {
    sv = (long *)my_malloc(sizeof(long)*(totdoc+11));
    for(i=0;i<totdoc;i++) 
      sv[i]=0;
    for(i=1;i<model->sv_num;i++) 
      if(a[model->supvec[i]->docnum] 
	 < (learn_parm->svm_cost[model->supvec[i]->docnum]
	    -learn_parm->epsilon_a)) {
	sv[model->supvec[i]->docnum]=1;
	svnum++;
      }
    sv2dnum = (long *)my_malloc(sizeof(long)*(totdoc+11));
    clear_index(sv2dnum);
    compute_index(sv,totdoc,sv2dnum);
  }

  for(i=0;i<totdoc;i++) {
    if(unlabeled[i]) {
      /* ignore it */
    }
    else {
      xi=1.0-((lin[i]-model->b)*(double)label[i]);
      if(xi<0) xi=0;
      if(label[i]>0) {
	totposex++;
      }
      if((learn_parm->rho*a[i]*r_delta_sq+xi) >= 1.0) {
	if(learn_parm->xa_depth > 0) {  /* makes assumptions */
	  sim=distribute_alpha_t_greedily(sv2dnum,svnum,docs,a,i,label,
					  kernel_parm,learn_parm,
		            (double)((1.0-xi-a[i]*r_delta_sq)/(2.0*a[i])));
	}
	if((learn_parm->xa_depth == 0) || 
	   ((a[i]*docs[i].twonorm_sq+a[i]*2.0*sim+xi) >= 1.0)) 
	{ 
	  looerror++;
	  if(label[i]>0) {
	    looposerror++;
	  }
	  else {
	    loonegerror++;
	  }
	}
      }
      totex++;
    }
  }

  (*error)=((double)looerror/(double)totex)*100.0;
  (*recall)=(1.0-(double)looposerror/(double)totposex)*100.0;
  (*precision)=(((double)totposex-(double)looposerror)
    /((double)totposex-(double)looposerror+(double)loonegerror))*100.0;

  free(sv);
  free(sv2dnum);
}


double CSVMLight::distribute_alpha_t_greedily(
long *sv2dnum, 
long svnum,
DOC *docs,       
double *a,
long docnum,
long *label,
KERNEL_PARM *kernel_parm,
LEARN_PARM *learn_parm,
double thresh)
{
  long best_depth=0;
  long i,j,k,d,skip,allskip;
  double best,best_val[101],val,init_val_sq,init_val_lin;
  long best_ex[101];
  CFLOAT *cache,*trow;

  cache=(CFLOAT *)my_malloc(sizeof(CFLOAT)*learn_parm->xa_depth*svnum);
  trow = (CFLOAT *)my_malloc(sizeof(CFLOAT)*svnum);

  for(k=0;k<svnum;k++) {
    trow[k]=kernel(kernel_parm,&(docs[docnum]),&(docs[sv2dnum[k]]));
  }

  init_val_sq=0;
  init_val_lin=0;
  best=0;

  for(d=0;d<learn_parm->xa_depth;d++) {
    allskip=1;
    if(d>=1) {
      init_val_sq+=cache[best_ex[d-1]+svnum*(d-1)]; 
      for(k=0;k<d-1;k++) {
        init_val_sq+=2.0*cache[best_ex[k]+svnum*(d-1)]; 
      }
      init_val_lin+=trow[best_ex[d-1]]; 
    }
    for(i=0;i<svnum;i++) {
      skip=0;
      if(sv2dnum[i] == docnum) skip=1;
      for(j=0;j<d;j++) {
	if(i == best_ex[j]) skip=1;
      }

      if(!skip) {
	val=init_val_sq;
	val+=docs[sv2dnum[i]].twonorm_sq;
	for(j=0;j<d;j++) {
	  val+=2.0*cache[i+j*svnum];
	}
	val*=(1.0/(2.0*(d+1.0)*(d+1.0)));
	val-=((init_val_lin+trow[i])/(d+1.0));

	if(allskip || (val < best_val[d])) {
	  best_val[d]=val;
	  best_ex[d]=i;
	}
	allskip=0;
	if(val < thresh) {
	  i=svnum;
	  /*	 CIO::message("EARLY"); */
	}
      }
    }
    if(!allskip) {
      for(k=0;k<svnum;k++) {
	  cache[d*svnum+k]=kernel(kernel_parm,
				  &(docs[sv2dnum[best_ex[d]]]),
				  &(docs[sv2dnum[k]]));
      }
    }
    if((!allskip) && ((best_val[d] < best) || (d == 0))) {
      best=best_val[d];
      best_depth=d;
    }
    if(allskip || (best < thresh)) {
      d=learn_parm->xa_depth;
    }
  }    

  free(cache);
  free(trow);

  /* CIO::message("Distribute[%ld](%ld)=%f, ",docnum,best_depth,best); */
  return(best);
}


/* loo-bound based on observation */
/* that loo-errors must have an   */
void CSVMLight::estimate_transduction_quality(MODEL *model, long *label, long* unlabeled, long totdoc, DOC *docs, double *lin)
{
  long i,j,l=0,ulab=0,lab=0,labpos=0,labneg=0,ulabpos=0,ulabneg=0,totulab=0;
  double totlab=0,totlabpos=0,totlabneg=0,labsum=0,ulabsum=0;
  double r_delta,r_delta_sq,xi,xisum=0,asum=0;

  r_delta=estimate_r_delta(docs,totdoc,&(model->kernel_parm)); 
  r_delta_sq=r_delta*r_delta;

  for(j=0;j<totdoc;j++) {
    if(unlabeled[j]) {
      totulab++;
    }
    else {
      totlab++;
      if(label[j] > 0) 
	totlabpos++;
      else 
	totlabneg++;
    }
  }
  for(j=1;j<model->sv_num;j++) {
    i=model->supvec[j]->docnum;
    xi=1.0-((lin[i]-model->b)*(double)label[i]);
    if(xi<0) xi=0;

    xisum+=xi;
    asum+=fabs(model->alpha[j]);
    if(unlabeled[i]) {
      ulabsum+=(fabs(model->alpha[j])*r_delta_sq+xi);
    }
    else {
      labsum+=(fabs(model->alpha[j])*r_delta_sq+xi);
    }
    if((fabs(model->alpha[j])*r_delta_sq+xi) >= 1) { 
      l++;
      if(unlabeled[model->supvec[j]->docnum]) {
	ulab++;
	if(model->alpha[j] > 0) 
	  ulabpos++;
	else 
	  ulabneg++;
      }
      else {
	lab++;
	if(model->alpha[j] > 0) 
	  labpos++;
	else 
	  labneg++;
      }
    }
  }
 CIO::message("xacrit>=1: labeledpos=%.5f labeledneg=%.5f default=%.5f\n",(double)labpos/(double)totlab*100.0,(double)labneg/(double)totlab*100.0,(double)totlabpos/(double)(totlab)*100.0);
 CIO::message("xacrit>=1: unlabelpos=%.5f unlabelneg=%.5f\n",(double)ulabpos/(double)totulab*100.0,(double)ulabneg/(double)totulab*100.0);
 CIO::message("xacrit>=1: labeled=%.5f unlabled=%.5f all=%.5f\n",(double)lab/(double)totlab*100.0,(double)ulab/(double)totulab*100.0,(double)l/(double)(totdoc)*100.0);
 CIO::message("xacritsum: labeled=%.5f unlabled=%.5f all=%.5f\n",(double)labsum/(double)totlab*100.0,(double)ulabsum/(double)totulab*100.0,(double)(labsum+ulabsum)/(double)(totdoc)*100.0);
 CIO::message("r_delta_sq=%.5f xisum=%.5f asum=%.5f\n",r_delta_sq,xisum,asum);
}

/* optional: length of model vector in feature space */
/* optional: radius of ball containing the data */
double CSVMLight::estimate_margin_vcdim(MODEL *model, double w, double R, KERNEL_PARM *kernel_parm)
{
  double h;

  /* follows chapter 5.6.4 in [Vapnik/95] */

  if(w<0) {
    w=model_length_s(model,kernel_parm);
  }
  if(R<0) {
    R=estimate_sphere(model,kernel_parm); 
  }
  h = w*w * R*R +1; 
  return(h);
}

/* Approximates the radius of the ball containing */
/* the support vectors by bounding it with the */
double CSVMLight::estimate_sphere(MODEL *model, KERNEL_PARM *kernel_parm)
{                         /* length of the longest support vector. This is */
  register long j;        /* pretty good for text categorization, since all */
  double xlen,maxxlen=0;  /* documents have feature vectors of length 1. It */
  DOC nulldoc;            /* assumes that the center of the ball is at the */
////  WORD nullword;          /* origin of the space. */

////  nullword.wnum=0;
////  nulldoc.words=&nullword;
  nulldoc.twonorm_sq=0;
  nulldoc.docnum=-1;

  for(j=1;j<model->sv_num;j++) {
    xlen=sqrt(kernel(kernel_parm,model->supvec[j],model->supvec[j])
	      -2*kernel(kernel_parm,model->supvec[j],&nulldoc)
	      +kernel(kernel_parm,&nulldoc,&nulldoc));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }

  return(maxxlen);
}

double CSVMLight::estimate_r_delta(DOC *docs, long totdoc, KERNEL_PARM *kernel_parm)
{
  long i;
  double maxxlen,xlen;
  DOC nulldoc;            /* assumes that the center of the ball is at the */
////  WORD nullword;          /* origin of the space. */

////  nullword.wnum=0;
////  nulldoc.words=&nullword;
  nulldoc.twonorm_sq=0;
  nulldoc.docnum=-1;

  maxxlen=0;
  for(i=0;i<totdoc;i++) {
    xlen=sqrt(kernel(kernel_parm,&(docs[i]),&(docs[i]))
	      -2*kernel(kernel_parm,&(docs[i]),&nulldoc)
	      +kernel(kernel_parm,&nulldoc,&nulldoc));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }

  return(maxxlen);
}

double CSVMLight::estimate_r_delta_average(DOC *docs, long totdoc, KERNEL_PARM *kernel_parm)
{
  long i;
  double avgxlen;
  DOC nulldoc;            /* assumes that the center of the ball is at the */
////  WORD nullword;          /* origin of the space. */

////  nullword.wnum=0;
////  nulldoc.words=&nullword;
  nulldoc.twonorm_sq=0;
  nulldoc.docnum=-1;

  avgxlen=0;
  for(i=0;i<totdoc;i++) {
    avgxlen+=sqrt(kernel(kernel_parm,&(docs[i]),&(docs[i]))
		  -2*kernel(kernel_parm,&(docs[i]),&nulldoc)
		  +kernel(kernel_parm,&nulldoc,&nulldoc));
  }

  return(avgxlen/totdoc);
}

double CSVMLight::length_of_longest_document_vector(DOC *docs, long totdoc, KERNEL_PARM *kernel_parm)
{
  long i;
  double maxxlen,xlen;

  maxxlen=0;
  for(i=0;i<totdoc;i++) {
    xlen=sqrt(kernel(kernel_parm,&(docs[i]),&(docs[i])));
    if(xlen>maxxlen) {
      maxxlen=xlen;
    }
  }

  return(maxxlen);
}

/****************************** IO-handling **********************************/

void CSVMLight::write_model(FILE *modelfl, MODEL *model)
{
  long i,j;

  if(verbosity>=1) {
   CIO::message("Writing model file...");
  }
  if (modelfl == NULL)
  { 
    perror ("modelfl not open"); 
    exit (1); 
  }
  fprintf(modelfl,"SVM-light Version %s\n",VERSION);
  fprintf(modelfl,"%ld # kernel type\n",
	  model->kernel_parm.kernel_type);
  /*fprintf(modelfl,"%ld # kernel parameter -d \n",
	  model->kernel_parm.poly_degree);
  fprintf(modelfl,"%.8g # kernel parameter -g \n",
	  model->kernel_parm.rbf_gamma);
  fprintf(modelfl,"%.8g # kernel parameter -s \n",
	  model->kernel_parm.coef_lin);
  fprintf(modelfl,"%.8g # kernel parameter -r \n",
	  model->kernel_parm.coef_const);*/
  fprintf(modelfl,"%10.16e # normalization (kernel parameter -u) \n",normalizer);
////  fprintf(modelfl,"%ld # highest feature index \n",model->totwords);
  fprintf(modelfl,"%ld # number of training documents \n",model->totdoc);
 
  fprintf(modelfl,"%ld # number of support vectors plus 1 \n",model->sv_num);
  fprintf(modelfl,"%.8g # threshold b \n",model->b);

  for(i=1;i<model->sv_num;i++) {
////    fprintf(modelfl,"%.32g ",model->alpha[i]);
    fprintf(modelfl,"%+10.16e:",model->alpha[i]);
	for (j=0;j<pos->get_observations()->get_obs_T((model->supvec[i])->docnum); j++)
		fprintf(modelfl, "%c", pos->get_observations()->remap(pos->get_observations()->get_obs((model->supvec[i])->docnum, j)));
////    for (j=0; ((model->supvec[i])->words[j]).wnum; j++) 
////	{
////      fprintf(modelfl,"%ld:%.8g ",
////	      (long)((model->supvec[i])->words[j]).wnum,
////	      (double)(((model->supvec[i])->words[j]).weight));
////    }
    fprintf(modelfl,"\n");
  }
  //fclose(modelfl);
  if(verbosity>=1) {
   CIO::message("done\n");
  }
}


void CSVMLight::write_prediction(char *predfile, MODEL *model, double *lin, double *a, 
					  long *unlabeled, long *label, long totdoc, LEARN_PARM *learn_parm)
{
  FILE *predfl;
  long i;
  double dist,a_max;

  if(verbosity>=1) {
   CIO::message("Writing prediction file...");
  }
  if ((predfl = fopen (predfile, "w")) == NULL)
  { perror (predfile); exit (1); }
  a_max=learn_parm->epsilon_a;
  for(i=0;i<totdoc;i++) {
    if((unlabeled[i]) && (a[i]>a_max)) {
      a_max=a[i];
    }
  }
  for(i=0;i<totdoc;i++) {
    if(unlabeled[i]) {
      if((a[i]>(learn_parm->epsilon_a))) {
	dist=(double)label[i]*(1.0-learn_parm->epsilon_crit-a[i]/(a_max*2.0));
      }
      else {
	dist=(lin[i]-model->b);
      }
      if(dist>0) {
	fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
      }
      else {
	fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
      }
    }
  }
  fclose(predfl);
  if(verbosity>=1) {
   CIO::message("done\n");
  }
}

void CSVMLight::write_alphas(char *alphafile, double *a, long *label, long totdoc)
{
  FILE *alphafl;
  long i;

  if(verbosity>=1) {
   CIO::message("Writing alpha file...");
  }
  if ((alphafl = fopen (alphafile, "w")) == NULL)
  { perror (alphafile); exit (1); }
  for(i=0;i<totdoc;i++) {
    fprintf(alphafl,"%.8g\n",a[i]*(double)label[i]);
  }
  fclose(alphafl);
  if(verbosity>=1) {
   CIO::message("done\n");
  }
}
