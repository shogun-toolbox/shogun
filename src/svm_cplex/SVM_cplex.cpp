#include "svm_cplex/SVM_cplex.h"
#include <matrix.h>
extern "C" {
#include "svm_cplex/train_svm.h"
}

extern long verbosity;
extern double normalizer;
extern CHMM* pos;
extern CHMM* neg;

CSVMCplex::CSVMCplex()
  :w(NULL),b(0)
{
}

CSVMCplex::~CSVMCplex()
{
  if (w!=NULL)
    free(w) ;
}

extern "C" void train_svm_main(double *XT, double *LT, double C, 
			       int ell, int dim, double *w, double *b) ;


bool CSVMCplex::svm_train(CObservation* train, int, double C)
{
  int number_of_examples = train->get_DIMENSION();
  int number_of_params   = 1+pos->get_N()*(2+pos->get_N()+pos->get_M())+neg->get_N()*(2+neg->get_N()+neg->get_M());
  fprintf(stderr, "SVM cplex ell=%i, d=%i\n", number_of_examples, number_of_params) ;
  
  double * XT=(double*)malloc(number_of_examples*number_of_params*sizeof(double)) ;
  double * LT=(double*)malloc(number_of_examples*sizeof(double)) ;
  
  fprintf(stderr, "generating features\n") ;
  for (int i=0; i<number_of_examples; i++)
    {
      top_feature(i, &XT[i*number_of_params]) ;
      LT[i]=train->get_label(i) ;
      if (i%100==0)
	fprintf(stderr, "%i..", i) ;
      /*	    fprintf(stderr, "LT[i]=%1.2e\n", LT[i]) ;*/
    } ;
  fprintf(stderr, "%i. Done.", number_of_examples) ;
  
  fprintf(stderr, "training SVM\n") ;
  if (w)
    free(w) ;
  w=(double*)malloc(number_of_params*sizeof(double)) ;
  train_svm_main(XT, LT, C, number_of_examples, number_of_params, w, &b) ;
  
  fprintf(stderr, "b=%e\n", b) ;
  /*for (int i=0; i<number_of_params; i++)
    fprintf(stderr, "w[i]=%1.2e\n", w[i]) ;*/
  
  /*
    for (int i=0; i<number_of_examples; i++) 
    {
      double out=b ;
      for (int j=0; j<number_of_params; j++)
	out+=XT[i*number_of_params+j]*w[j] ;
      fprintf(stderr,"label[%i]=%i, out[%i]=%1.2f\n", i, train->get_label(i), i, out) ;
    } ;
  */
  free(XT) ;
  free(LT) ;
  
  return true ;
}

bool CSVMCplex::svm_test(CObservation* test, FILE* outfile, FILE* rocfile)
{
  int number_of_examples=test->get_DIMENSION();
  int number_of_params   = 1+pos->get_N()*(2+pos->get_N()+pos->get_M())+neg->get_N()*(2+neg->get_N()+neg->get_M());
  
  double *output = new double[number_of_examples];	
  int* label     = new int[number_of_examples];	
  double *XT     = (double*)malloc(number_of_params*sizeof(double)) ;

  for (int i=0; i<number_of_examples; i++) 
    {
      top_feature(i, XT) ;
      {
	double out=b ;
	for (int j=0; j<number_of_params; j++)
	  out+=XT[j]*w[j] ;
	output[i]=out ;
      } 
      label[i]=test->get_label(i) ;
      if ((label[i] < 0 && output[i] < 0) || (label[i] > 0 && output[i] > 0))
	fprintf(outfile,"%+.8g (%+d)\n",output[i], label[i]);
      else
	fprintf(outfile,"%+.8g (%+d)(*)\n",output[i], label[i]);
    } ;
  
  double *fp= new double[number_of_examples];	
  double *tp= new double[number_of_examples];	
  int possize=-1;
  int negsize=-1;
  
  int pointeven=math.calcroc(fp, tp, output, label, number_of_examples, possize, negsize, rocfile);
  
  double correct=possize*tp[pointeven]+(1-fp[pointeven])*negsize;
  double fpo=fp[pointeven]*negsize;
  double fne=(1-tp[pointeven])*possize;
  
  printf("classified:\n");
  printf("\tcorrect:%i\n", int (correct));
  printf("\twrong:%i (fp:%i,fn:%i)\n", int(fpo+fne), int (fpo), int (fne));
  printf("of %i samples (c:%f,w:%f,fp:%f,tp:%f)\n",number_of_examples, correct/number_of_examples, 1-correct/number_of_examples, fp[pointeven], tp[pointeven]);

  delete[] fp;
  delete[] tp;
  delete[] output;
  delete[] label;
  return true;
}

bool CSVMCplex::load_svm(FILE* modelfl, CObservation*)
{
    bool result=true;
    char version_buffer[1024];

    fscanf(modelfl,"SVM-cplex Version %s\n",version_buffer);
    if(strcmp(version_buffer,SVM_CPLEX_VERSION)) {
	perror ("Version of model-file does not match version of SVM_cplex!"); 
	exit (1); 
    }
    fscanf(modelfl,"%le\n",&b);
    fprintf(stderr, "b=%e\n", b) ;
    int number_of_params2;
    fscanf(modelfl,"%i\n",&number_of_params2);
    int number_of_params   = 1+pos->get_N()*(2+pos->get_N()+pos->get_M())+neg->get_N()*(2+neg->get_N()+neg->get_M());
    if (number_of_params!=number_of_params)
      {
	perror("number of parameters do not match") ;
	exit(1) ;
      } ;
    if (w)
      free(w) ;
    w=(double*)malloc(number_of_params*sizeof(double)) ;
    for (int i=0; i<number_of_params; i++)
      {
	double d ;
	fscanf(modelfl,"%le\n", &d) ;
	w[i]=d ;
	/*fprintf(stderr, "w[i]=%e\n", w[i]) ;*/
      } ;
    return result ;
}

bool CSVMCplex::save_svm(FILE* modelfl)
{
    bool result=true;
    
    int number_of_params   = 1+pos->get_N()*(2+pos->get_N()+pos->get_M())+neg->get_N()*(2+neg->get_N()+neg->get_M());

    fprintf(modelfl,"SVM-cplex Version %s\n",SVM_CPLEX_VERSION);
    fprintf(modelfl,"%e\n", b) ;
    fprintf(modelfl,"%i\n", number_of_params) ;
    for (int i=0; i<number_of_params; i++)
      fprintf(modelfl,"%e\n", w[i]) ;

    return result ;
}

void CSVMCplex::top_feature(int x, double *feat)
{
  int i,j;
  
  double posx=pos->model_probability(x);
  double negx=neg->model_probability(x);
  
  feat[0]=posx-negx ;
      
  int p=1;
  for (i=0; i<pos->get_N(); i++)
    {
      feat[p++]=exp(pos->model_derivative_p(i, x)-posx);
      feat[p++]=exp(pos->model_derivative_q(i, x)-posx);
      
      for (j=0; j<pos->get_N(); j++)
	feat[p++]=exp(pos->model_derivative_a(i, j, x)-posx);
      
      for (j=0; j<pos->get_M(); j++)
	feat[p++]=exp(pos->model_derivative_b(i, j, x)-posx);
    }

  for (i=0; i<neg->get_N(); i++)
    {
      feat[p++]=exp(neg->model_derivative_p(i, x)-negx);
      feat[p++]=exp(neg->model_derivative_q(i, x)-negx);
      
      for (j=0; j<neg->get_N(); j++)
	feat[p++]=exp(neg->model_derivative_a(i, j, x)-negx);
      
      for (j=0; j<neg->get_M(); j++)
	feat[p++]=exp(neg->model_derivative_b(i, j, x)-negx);
    }
}



