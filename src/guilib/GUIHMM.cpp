#include <unistd.h>
#include "lib/common.h"
#include "guilib/GUIHMM.h"
#include "hmm/Observation.h"
#include "gui/GUI.h"

CGUIHMM::CGUIHMM(CGUI * gui_): gui(gui_)
{
	working=NULL;

	pos=NULL;
	neg=NULL;
	test=NULL;

	ITERATIONS=150;
	EPSILON=1e-4;
	PSEUDO=1e-10;
	M=4;
	ORDER=1;

	conv_it=5;
}

CGUIHMM::~CGUIHMM()
{

}

bool CGUIHMM::new_hmm(char* param)
{
	param=CIO::skip_spaces(param);

	int n,m,order;
	if (sscanf(param, "%d %d %d", &n, &m, &order) == 3)
	{
	  if (working)
	    delete working;
	 
	  if (order>1)
	     CIO::message("WARNING: no order > 1 supported\n"); 

	  working=new CHMM(n,m,order,NULL,PSEUDO);
	  ORDER=order;
	  M=m;
	  return true;
	}
	else
	  CIO::message("see help for parameters\n");

	return false;
}

bool CGUIHMM::baum_welch_train(char* param)
{
	char templname[]=TMP_DIR "bw_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working);
			
			double prob_train=math.ALMOST_NEG_INFTY, prob = -math.INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_baum_welch(working_estimate);
				prob_train=working_estimate->model_probability();
				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message("\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message("\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message("assign observation first\n");
	}
	else
		CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::baum_welch_train_defined(char* param)
{
	char templname[]=TMP_DIR "bwdef_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working);
			
			double prob_train=math.ALMOST_NEG_INFTY, prob = -math.INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_baum_welch_defined(working_estimate);
				prob_train=working_estimate->model_probability();
				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message("\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message("\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message("assign observation first\n");
	}
	else
		CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::viterbi_train(char* param)
{
	char* templname= TMP_DIR "vit_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working);
			
			double prob_train=math.ALMOST_NEG_INFTY, prob = -math.INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_viterbi(working_estimate);
				prob_train=working_estimate->best_path(-1);

				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message("\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message("\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message("assign observation first\n");
	}
	else
		CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::viterbi_train_defined(char* param)
{
	char* templname= TMP_DIR "vitdef_model_XXXXXX" ;
#ifdef SUNOS
#define mkstemp(name) mktemp(name);
#endif
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;

	if (working) 
	{
		if (working->get_observations())
		{
			CHMM* working_estimate=new CHMM(working);
			
			double prob_train=math.ALMOST_NEG_INFTY, prob = -math.INFTY ;

			while (!converge(prob,prob_train))
			{
				switch_model(&working, &working_estimate);
				prob=prob_train ;
				working->estimate_model_viterbi_defined(working_estimate);
				prob_train=working_estimate->best_path(-1);

				if (prob_max<prob_train)
				{
					prob_max=prob_train ;
#ifdef TMP_SAVE
					FILE* file=fopen(templname_best, "w");
					CIO::message("\nsaving best model with filename %s ... ", templname_best) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				} 
				else
				{
#ifdef TMP_SAVE
					FILE* file=fopen(templname, "w");
					CIO::message("\nsaving model with filename %s ... ", templname) ;
					working->save_model(file) ;
					fclose(file) ;
					CIO::message("done.") ;
#endif
				}
			}
			delete working_estimate;
			working_estimate=NULL;
		}
		else
			CIO::message("assign observation first\n");
	}
	else
		CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::linear_train(char* param)
{
    if (working) 
    {
	if (working->get_observations())
	{
	    working->linear_train();
	    return true;
	}
	else
	    CIO::message("assign observation first\n");
    }
    else
	CIO::message("create model first\n");

    return false;
}

bool CGUIHMM::linear_train_from_file(char* param)
{
    bool result=false;
    E_OBS_ALPHABET alphabet;
    int WIDTH=-1,UPTO=-1;
    char fname[1024];

    param=CIO::skip_spaces(param);
    sscanf(param, "%s %d %d", fname, &WIDTH, &UPTO);

    FILE* file=fopen(fname, "r");

    if (file) 
    {
	if (WIDTH < 0 || UPTO < 0 )
	{
	    int i=0;

	    while (fgetc(file)!='\n' && !feof(file))
		i++;

	    if (!feof(file))
	    {
		WIDTH=i+1;
		UPTO=i;
		CIO::message("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
		fseek(file,0,SEEK_SET);
	    }
	    else
		return false;
	}

	if (WIDTH >0 && UPTO >0)
	{	  
	    alphabet=DNA;
	    //ORDER=1; //obsoleted by set_order
	    M=4;

	    CObservation* obs=new CObservation(TRAIN, alphabet, (BYTE)ceil(log(M)/log(2)), M, ORDER);

	    if (working && obs)
	    {
		alphabet=obs->get_alphabet();
		ORDER=working->get_ORDER();
		delete(working);
		working=NULL;

		switch (alphabet)
		{
		    case DNA:
			M=4;
			break;
		    case PROTEIN:
			M=26;
			break;
		    case CUBE:
			M=6;
			break;
		    case ALPHANUM:
			M=36;
			break;
		    default:
			M=4;
			break;
		};
	    }

	    working=new CHMM(UPTO,M,ORDER,NULL,PSEUDO);

	    if (working)
	    {
		working->set_observation_nocache(obs);
		working->linear_train(file, WIDTH, UPTO);
		result=true;
		CIO::message("done.\n");
	    }
	    else
		CIO::message("model creation failed\n");

	    delete obs;
	}

	fclose(file);
    }
    else
	CIO::message("opening file %s failed!\n", fname);

    return result;
}

bool CGUIHMM::one_class_test(char* param)
{
	bool result=false;
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;
	double tresh=0;
	int linear=0;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%le %s %s %d", &tresh, outputname, rocfname, &linear);
	CIO::message("Tresholding at:%f\n",tresh);

	if (numargs>=2)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(stderr,"ERROR: could not open %s\n",outputname);
			return false;
		}

		if (numargs==3) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
				return false;
			}
		}
	}

	if (test)
	{
		if (gui->guiobs.get_obs("POSTEST") && gui->guiobs.get_obs("NEGTEST"))
		{
			CObservation* obs=new CObservation(gui->guiobs.get_obs("POSTEST"), gui->guiobs.get_obs("NEGTEST"));

			CObservation* old_test=test->get_observations();
			test->set_observations(obs);

			int total=obs->get_DIMENSION();

			REAL* output = new REAL[total];	
			int* label= new int[total];	

			for (int dim=0; dim<total; dim++)
			{
				output[dim]= linear ? test->linear_model_probability(dim) : test->model_probability(dim);
				label[dim]= obs->get_label(dim);
			}

			gui->guimath.evaluate_results(output, label, total, tresh, outputfile, rocfile);
			delete[] output;
			delete[] label;

			test->set_observations(old_test);

			delete obs;

			result=true;
		}
		else
			CIO::message("assign posttest and negtest observations first!\n");
	}
	else
		CIO::message("assign test model first!\n");

	if (rocfile)
		fclose(rocfile);
	if (outputfile)
		fclose(outputfile);
	return result;
}

bool CGUIHMM::test_hmm(char* param)
{
	bool result=false;
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;
	int poslinear=0;
	int neglinear=0;

	double tresh=0;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%le %s %s %d %d", &tresh, outputname, rocfname, &neglinear, &poslinear);
	CIO::message("Tresholding at:%f\n",tresh);

	if (numargs>=2)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(stderr,"ERROR: could not open %s\n",outputname);
			return false;
		}

		if (numargs==3) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
				return false;
			}
		}
	}

	if (pos && neg)
	{
		if (gui->guiobs.get_obs("POSTEST") && gui->guiobs.get_obs("NEGTEST"))
		{
			CObservation* obs=new CObservation(gui->guiobs.get_obs("POSTEST"), gui->guiobs.get_obs("NEGTEST"));

			CObservation* old_pos=pos->get_observations();
			CObservation* old_neg=neg->get_observations();

			pos->set_observations(obs);
			neg->set_observations(obs);

			int total=obs->get_DIMENSION();

			REAL* output = new REAL[total];	
			int* label= new int[total];	
			
			CIO::message("testing using neg %s hmm vs. pos %s hmm\n", neglinear ? "linear" : "", poslinear ? "linear" : "");

			for (int dim=0; dim<total; dim++)
			{
				output[dim]= 
				    (poslinear ? pos->linear_model_probability(dim) : pos->model_probability(dim)) -
				    (neglinear ? neg->linear_model_probability(dim) : neg->model_probability(dim));
				label[dim]= obs->get_label(dim);
			}
			
			gui->guimath.evaluate_results(output, label, total, tresh, outputfile, rocfile);

			delete[] output;
			delete[] label;

			pos->set_observations(old_pos);
			neg->set_observations(old_neg);

			delete obs;
			result=true;
		}
		else
			printf("assign postest and negtest observations first!\n");
	}
	else
		CIO::message("assign positive and negative models first!\n");

	if (rocfile)
		fclose(rocfile);
	if (outputfile)
		fclose(outputfile);

	return result;
}

bool CGUIHMM::append_model(char* param)
{
	if (working)
	{
		char fname[1024]; 
		int base1=0;
		int base2=2;
		param=CIO::skip_spaces(param);

		int num_param=sscanf(param, "%s %i %i", fname, &base1, &base2);

		if (num_param==3 || num_param==1)
		{
			FILE* model_file=fopen(fname, "r");

			if (model_file)
			{
				CHMM* h=new CHMM(model_file,PSEUDO);
				if (h && h->get_status())
				{
					printf("file successfully read\n");
					fclose(model_file);

					REAL cur_o[4];
					REAL app_o[4];

					for (int i=0; i<h->get_M(); i++)
					{
						if (i==base1)
							cur_o[i]=0;
						else
							cur_o[i]=-1000;
						
						if (i==base2)
							app_o[i]=0;
						else
							app_o[i]=-1000;
					}
					
					if (num_param==3)
					    working->append_model(h, cur_o, app_o);
					else
					    working->append_model(h);
					CIO::message("new model has %i states\n", working->get_N());
					delete h;
				}
				else
					CIO::message("reading file %s failed\n", fname);
			}
			else
				CIO::message("opening file %s failed\n", fname);
		}
		else
			CIO::message("see help for parameters\n", fname);
	}
	else
		CIO::message("create model first\n");


	return false;
}

bool CGUIHMM::add_states(char* param)
{
	if (working)
	{
		int states=1;
		double value=0;
  
		param=CIO::skip_spaces(param);

		sscanf(param, "%i %le", &states, &value);
		CIO::message("adding %i states\n", states);
		working->add_states(states, value);
		CIO::message("new model has %i states\n", working->get_N());
		return true;
	}
	else
	   CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::set_pseudo(char* param)
{
  param=CIO::skip_spaces(param);
  
  if (sscanf(param, "%le", &PSEUDO)!=1)
    {
      CIO::message("see help for parameters. current setting: pseudo=%e\n", PSEUDO);
      return false ;
    }
  CIO::message("current setting: pseudo=%e\n", PSEUDO);
  return true ;
}

bool CGUIHMM::convergence_criteria(char* param)
{
  int j=100;
  double f=0.001;
  
  param=CIO::skip_spaces(param);
  
  if (sscanf(param, "%d %le", &j, &f) == 2)
    {
      ITERATIONS=j;
      EPSILON=f;
    }
  else
    {
      CIO::message("see help for parameters. current setting: iterations=%i, epsilon=%e\n",ITERATIONS,EPSILON);
      return false ;
    }
  CIO::message("current setting: iterations=%i, epsilon=%e\n",ITERATIONS,EPSILON);
  return true ;
} ;

bool CGUIHMM::set_hmm_as(char* param)
{
	param=CIO::skip_spaces(param);
	char target[1024];

	if ((sscanf(param, "%s", target))==1)
	{
		if (working)
		{
			if (strcmp(target,"POS")==0)
			{
				delete pos;
				pos=working;
				working=NULL;
			}
			else if (strcmp(target,"NEG")==0)
			{
				delete neg;
				neg=working;
				working=NULL;
			}
			else if (strcmp(target,"TEST")==0)
			{
				delete test;
				test=working;
				working=NULL;
			}
			else
				CIO::message("target POS|NEG|TEST missing\n");
		}
		else
			CIO::message("create model first!\n");
	}
	else
		CIO::message("target POS|NEG|TEST missing\n");

	return false;
}

bool CGUIHMM::assign_obs(char* param)
{
  param=CIO::skip_spaces(param);
  
  char target[1024];
  
  if ((sscanf(param, "%s", target))==1)
    {
      if (working)
	{
	  CObservation *obs=gui->guiobs.get_obs(target) ;
	  working->set_observations(obs);

	  return true ;
	}
      else
	{
	  printf("create model first!\n");
	  return false ;
	} ;
    }
  else
    printf("target POSTRAIN|NEGTRAIN|POSTEST|NEGTEST|TEST missing\n");
  return false ;
} ;

//convergence criteria  -tobeadjusted-
bool CGUIHMM::converge(double x, double y)
{
    double diff=y-x;
    double absdiff=fabs(diff);

    CIO::message("\n #%03d\tbest result so far: %G (eps: %f", iteration_count, y, diff);
    if (diff<0.0)
	//CIO::message(" ***") ;
	CIO::message(" **************** WARNING **************") ;
    CIO::message(")") ;

    if (iteration_count-- == 0 || (absdiff<EPSILON && conv_it<=0))
    {
	iteration_count=ITERATIONS;
	CIO::message("...finished\n");
	conv_it=5 ;
	return true;
    }
    else
    {
	if (absdiff<EPSILON)
	    conv_it-- ;
	else
	    conv_it=5;

	return false;
    }
}

//switch model and train model
void CGUIHMM::switch_model(CHMM** m1, CHMM** m2)
{
    CHMM* dummy= *m1;

    *m1= *m2;
    *m2= dummy;
}

bool CGUIHMM::load(char* param)
{
	bool result=false;

	param=CIO::skip_spaces(param);

	if (working)
		delete working;
	working=NULL;

	FILE* model_file=fopen(param, "r");

	if (model_file)
	{
		working=new CHMM(model_file,PSEUDO);
		rewind(model_file);

		if (working && working->get_status())
		{
			printf("file successfully read\n");
			result=true;
		}

		ORDER=working->get_ORDER();
		M=working->get_M();
		fclose(model_file);
	}
	else
		CIO::message("opening file %s failed\n", param);

	return result;
}

bool CGUIHMM::save(char* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	char fname[1024];
	int binary=0;

	if (working)
	{
		if (sscanf(param, "%s %d", fname, &binary) >= 1)
		{
			FILE* file=fopen(fname, "w");
			if (file)
			{
				if (binary)
					result=working->save_model_bin(file);
				else
					result=working->save_model(file);
			}

			if (!file || !result)
				printf("writing to file %s failed!\n", fname);
			else
				printf("successfully written model into \"%s\" !\n", fname);

			if (file)
				fclose(file);
		}
		else
			CIO::message("see help for parameters\n");
	}
	else
		CIO::message("create model first\n");

	return result;
}

bool CGUIHMM::load_defs(char* param)
{
	param=CIO::skip_spaces(param);
	char fname[1024];
	int init=1;

	if (working)
	{
		if (sscanf(param, "%s %d", fname, &init) >= 1)
		{
			FILE* def_file=fopen(fname, "r");
			if (def_file && working->load_definitions(def_file,true,(init!=0)))
			{
				CIO::message("file successfully read\n");
				return true;
			}
			else
				CIO::message("opening file %s failed\n", fname);
		}
		else
			CIO::message("see help for parameters\n");
	}
	else
		CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::save_path(char* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	char fname[1024];
	int binary=0;

	if (working)
	{
		if (sscanf(param, "%s %d", fname, &binary) >= 1)
		{
			FILE* file=fopen(fname, "w");
			if (file)
			{
				/// ..future
				//if (binary)
				//	result=working->save_model_bin(file);
				//else
					
				result=working->save_path(file);
			}

			if (!file || !result)
				printf("writing to file %s failed!\n", fname);
			else
				printf("successfully written path into \"%s\" !\n", fname);

			if (file)
				fclose(file);
		}
		else
			CIO::message("see help for parameters\n");
	}
	else
		CIO::message("create model first\n");

	return result;
}

bool CGUIHMM::chop(char* param)
{
	param=CIO::skip_spaces(param);
	double value;

	if (sscanf(param, "%le", &value) == 1)
	{
	    if (working)
			working->chop(value);
		return true;
	}
	else
	   CIO::message("see help for parameters/create model first\n");
	return false;
}

bool CGUIHMM::likelihood(char* param)
{
	if (working)
	{
		working->output_model(false);
		return true;
	}
	else
		CIO::message("create model first!\n");
	return false;
}

bool CGUIHMM::output_hmm(char* param)
{
	if (working)
	{
		working->output_model(true);
		return true;
	}
	else
		CIO::message("create model first!\n");
	return false;
}

bool CGUIHMM::output_hmm_defined(char* param)
{
	if (working)
	{
		working->output_model_defined(true);
		return true;
	}
	else
		CIO::message("create model first!\n");
	return false;
}


bool CGUIHMM::best_path(char* param)
{
	if (working)
	{
	    working->output_model_sequence(false);
		return true;
	}
	else
	   CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::normalize(char* param)
{
	if (working)
	{
	    working->normalize();
		return true;
	}
	else
	   CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::output_hmm_path(char* param)
{
	param=CIO::skip_spaces(param);
	int from, to;

	if (sscanf(param, "%d %d", &from, &to) != 2)
	{
	    from=0; 
	    to=10 ;
	}

	if (working)
	{
	    working->output_model_sequence(true,from,to);
		return true;
	}
	else
	   CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::relative_entropy(char* param)
{
	if (pos && neg) 
	{
		if ( (pos->get_M() == neg->get_M()) && (pos->get_N() == neg->get_N()) )
		{
			double* entropy=new double[pos->get_N()];
			double* p=new double[pos->get_M()];
			double* q=new double[pos->get_M()];

			for (int i=0; i<pos->get_N(); i++)
			{
				for (int j=0; j<pos->get_M(); j++)
				{
					p[j]=pos->get_b(i,j);
					q[j]=neg->get_b(i,j);
				}

				entropy[i]=math.relative_entropy(p, q, pos->get_M());
				CIO::message("%f ", entropy[i]);
			}
			CIO::message("\n");
#error todo save me
			delete[] p;
			delete[] q;
			delete[] entropy;
		}
		else
			CIO::message("pos and neg hmm's differ in number of emissions or states\n");
	}
	else
		CIO::message("set pos and neg hmm first\n");
	return false;
}

bool CGUIHMM::entropy(char* param)
{
	if (pos) 
	{
		double* entropy=new double[pos->get_N()];
		double* p=new double[pos->get_M()];

		for (int i=0; i<pos->get_N(); i++)
		{
			for (int j=0; j<pos->get_M(); j++)
			{
				p[j]=pos->get_b(i,j);
			}

			entropy[i]=math.entropy(p, pos->get_M());
			CIO::message("%f ", entropy[i]);
		}
		CIO::message("\n");

#error todo save me
		delete[] p;
		delete[] entropy;
	}
	else
		CIO::message("set pos hmm first\n");
	return false;
}
