#include "guilib/GUISVM.h"
#include "gui/GUI.h"
#include "lib/io.h"

CGUISVM::CGUISVM(CGUI * gui_)
  : gui(gui_), svm_light(), 
#ifdef SVMMPI
#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
svm_mpi(gui->argc, gui->argv),
#endif
#endif // SVMMPI
#ifdef SVMCPLEX
  svm_cplex(),
#endif // SVMCPLEX
  svm(&svm_mpi)
{
}

CGUISVM::~CGUISVM()
{
}

bool CGUISVM::new_svm(char* param)
{
  CIO::not_implemented() ; // verstehe nicht, was hier hin soll ... ?-)
  return false ;
}

bool CGUISVM::train(char* param)
{
  if (!svm)
    {
      CIO::message("no svm available") ;
      return false ;
    } ;
  if (!gui->guifeatures.get_train_features())
    {
      CIO::message("no training features available") ;
      return false ;
    } ;
  
  CFeatures * f=gui->guifeatures.get_train_features() ;
  CIO::message("starting svm\n") ;
  return svm->svm_train(f) ;
}

bool CGUISVM::test(char* param)
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::set_svm_type(char* param)
{
  if (strcmp(param,"light"))
    svm=&svm_light ;
  else if (strcmp(param,"cplex"))
#ifdef SVMCPLEX
    svm=&svm_cplex ;
#else
  CIO::message("CPLEX SVM disabled\n") ;
#endif
  else if (strcmp(param,"mpi"))
#ifdef SVMMPI
#if defined(HAVE_MPI) && !defined(DISABLE_MPI)
    svm=&svm_mpi ;
#endif
#else
  CIO::message("MPI SVM disabled\n") ;
#endif
  
  return false ;
}

bool CGUISVM::get_svm_type()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::set_kernel()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::get_kernel()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::set_preproc()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::get_preproc()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::load_svm()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::save_svm()
{
  CIO::not_implemented() ;
  return false ;
}

bool CGUISVM::set_C(char* param)
{
  sscanf(param, "%le", &C) ;
  CIO::message("Set to C=%f\n", C) ;
  return true ;  
}



#if 0
#include "svm/SVM.h"
#include "svm/SVM_light.h"
#ifdef SVMCPLEX
 #include "svm_cplex/SVM_cplex.h"
#endif
#ifdef SVMCPLEX
#include <libmmfile.h>
#endif // SVMCPLEX



/*#include "guilib/SVMFunctions.h"

void CSVMFunctions::train_svm()
{
	delete[] docs;
	delete[] label;
	docs=NULL;
	label=NULL;
	delete[] mymodel.supvec;
	delete[] mymodel.alpha;
	delete[] mymodel.index;
	memset(&mymodel, 0x0, sizeof(MODEL));
	memset(&mymodel, 0x0, sizeof(MODEL));
	memset(&mykernel_cache, 0x0, sizeof(KERNEL_CACHE));
	memset(&mylearn_parm, 0x0, sizeof(LEARN_PARM));
	memset(&mykernel_parm, 0x0, sizeof(KERNEL_PARM));

	int totdoc=train->get_DIMENSION();

	docs=new DOC[totdoc];  // training examples
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
	mylearn_parm.epsilon_crit=5E-4;
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

void CSVMFunctions::test_svm()
{
    DOC doc;   // test example

    int total=test->get_DIMENSION();
    
    if (mymodel.kernel_parm.kernel_type==4) // standard hmm+svm
    {
	if (CHMM::compute_top_feature_cache(pos, neg))
	    mymodel.kernel_parm.kernel_type=6; // hmm+svm precalculated
    }

    REAL* output = new REAL[total];	
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

    REAL* fp= new REAL[total];	
    REAL* tp= new REAL[total];	
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
  //    free(mymodel.alpha);
}

void CSVMFunctions::load_svm()
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

void CSVMFunctions::save_svm()
{
	svm.save_svm(file);
}*/
#endif
