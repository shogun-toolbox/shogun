#include "svm/SVM.h"
#include "svm/svm_common.h"
#include "svm/svm_learn.h"
#include "svm/kernel.h"

extern long verbosity;
extern double normalizer;

CSVM::CSVM()
{
}

CSVM::~CSVM()
{
}

bool CSVM::svm_train(char* svm, CObservation* train, int kernel_type)
{
	int totdoc=train->get_DIMENSION();

	DOC* docs=new DOC[totdoc];  /* training examples */
	long* label=new long[totdoc];
	char str[1024];

	KERNEL_CACHE kernel_cache;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	MODEL model;

	for (int i=0; i<totdoc; i++)
	{
		docs[i].docnum=i;
		docs[i].twonorm_sq=0;
		label[i]=train->get_label(i);
		//printf("%i -> %i\n", (int) docs[i].docnum, (int) label[i]);
	}

	verbosity=1;
	strcpy (learn_parm.predfile, svm);
	strcpy (learn_parm.alphafile, "");
	learn_parm.biased_hyperplane=1;
	learn_parm.remove_inconsistent=0;
	learn_parm.skip_final_opt_check=1;
	learn_parm.svm_maxqpsize=8;
	learn_parm.svm_newvarsinqp=0;
	learn_parm.svm_iter_to_shrink=100;
	learn_parm.svm_c=1;
	learn_parm.transduction_posratio=0.5;
	learn_parm.svm_costratio=1.0;
	learn_parm.svm_costratio_unlab=1.0;
	learn_parm.svm_unlabbound=1E-5;
	learn_parm.epsilon_crit=0.00001;
	learn_parm.epsilon_a=1E-15;
	learn_parm.compute_loo=0;
	learn_parm.rho=1.0;
	learn_parm.xa_depth=0;
	kernel_parm.kernel_type=kernel_type; //custom kernel
	kernel_parm.poly_degree=-12345;
	kernel_parm.rbf_gamma=-12345;
	kernel_parm.coef_lin=-12345;
	kernel_parm.coef_const=-12345;
	
	//tester(); //to check kernel
	double norm_val=find_normalizer(&kernel_parm, totdoc);

	sprintf(str,"%.32g",norm_val);
	strcpy(kernel_parm.custom, str);

	kernel_cache_init(&kernel_cache,totdoc,100);
	svm_learn(docs,label,totdoc,-12345,&learn_parm,&kernel_parm,&kernel_cache,&model);
	kernel_cache_cleanup(&kernel_cache);
	write_model(svm,&model);

	free(model.supvec);
	free(model.alpha);
	free(model.index);
	delete[] docs;
	delete[] label;

	return true;
}

bool CSVM::svm_test(CObservation* test, FILE* outfile)
{
    DOC doc;   // test example

    int total=test->get_DIMENSION();

    double *output = new double[total];	
    int* label= new int[total];	

    for (int i=0; i<total; i++)
    { 
	doc.docnum=i;
	doc.twonorm_sq=-1;
	output[i]=classify_example(&svm,&doc);

	label[i]=test->get_label(i);
	if ((label[i] < 0 && output[i] < 0) || (label[i] > 0 && output[i] > 0))
	    fprintf(outfile,"%+.8g (%+d)\n",output[i], label[i]);
	else
	    fprintf(outfile,"%+.8g (%+d)(*)\n",output[i], label[i]);
    }  

    double *fp= new double[total];	
    double *tp= new double[total];	
    int possize=-1;
    int negsize=-1;

    int pointeven=math.calcroc(fp, tp, output, label, total, possize, negsize);

    double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
    double fpo=fp[pointeven]*negsize;
    double fne=(1-tp[pointeven])*possize;

    printf("classified:\n");
    printf("\tcorrect:%i\n", int (correct));
    printf("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
    printf("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",total, correct/total, 1-correct/total, fp[pointeven], tp[pointeven]);
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
      dist=classify_example(&model,&doc);
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
	printf("%ld..",totdoc); fflush(stdout);
      }
    }
  }  
  free(line);
  free(doc.words);
  free(model.supvec);
  free(model.alpha);
*/
}

bool CSVM::load_svm(FILE* modelfl, CObservation* test)
{
    bool result=false;
    char version_buffer[1024];
    MODEL* model=&svm;

    fscanf(modelfl,"SVM-light Version %s\n",version_buffer);
    if(strcmp(version_buffer,VERSION)) {
	perror ("Version of model-file does not match version of svm_classify!"); 
	exit (1); 
    }

    fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.kernel_type);  
//    fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.poly_degree);
//    fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.rbf_gamma);
//    fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_lin);
//    fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_const);
    fscanf(modelfl,"%lf%*[^\n]\n", &normalizer);

    ////  fscanf(modelfl,"%ld%*[^\n]\n", &model->totwords);
    fscanf(modelfl,"%ld%*[^\n]\n", &model->totdoc);
    fscanf(modelfl,"%ld%*[^\n]\n", &model->sv_num);
    fscanf(modelfl,"%lf%*[^\n]\n", &model->b);
    int file_pos=ftell(modelfl);
    
    model->sv_num--;
    printf("loading %ld support vectors\n",model->sv_num);
    test->add_support_vectors(modelfl, model->sv_num);
    fseek(modelfl, file_pos, SEEK_SET);

    model->supvec=new DOC*[model->sv_num];
    model->alpha=new double[model->sv_num];

    for (int i=0; i<model->sv_num; i++)
    {
	model->supvec[i] = new DOC;
	model->supvec[i]->docnum=test->get_support_vector_idx(i);
	model->supvec[i]->twonorm_sq=-1;
	fscanf(modelfl,"%lf%*[^\n]\n",&model->alpha[i]);
#ifdef DEBUG
	printf("alpha:%e,idx:%d\n",model->alpha[i],model->supvec[i]->docnum);
#endif
    }

    result=true;
    svm_loaded=result;
    return result;
}

