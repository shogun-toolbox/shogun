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
	char templname[35]=TMP_DIR "bw_model_XXXXXX" ;
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

bool CGUIHMM::linear_train(char* param)
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
			char buf[1024];
			if ( (fread(buf, sizeof (unsigned char), sizeof(buf), file)) == sizeof(buf))
			{
				for (int i=0; i<(int)sizeof(buf); i++)
				{
					if (buf[i]=='\n')
					{
						WIDTH=i+1;
						UPTO=i;
						CIO::message("detected WIDTH=%d UPTO=%d\n",WIDTH, UPTO);
						break;
					}
				}

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

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%le %s %s", &tresh, outputname, rocfname);
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
				output[dim]=test->model_probability(dim);
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

	double tresh=0;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%le %s %s", &tresh, outputname, rocfname);
	CIO::message("Tresholding at:%f\n",tresh);

	if (numargs>=2)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(stderr,"ERROR: could not open %s\n",outputname);
			return false;
		}

		if (numargs==2) 
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

			for (int dim=0; dim<total; dim++)
			{
				output[dim]=pos->model_probability(dim)-neg->model_probability(dim);
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

		if (sscanf(param, "%s %i %i", fname, &base1, &base2) == 3)
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
					
					working->append_model(h, cur_o, app_o);
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

	if (working)
	{
		FILE* file=fopen(param, "w");

		if ((!file) ||	(!working->save_model(file)))
			printf("writing to file %s failed!\n", param);
		else
		{
			printf("successfully written model into \"%s\" !\n", param);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		CIO::message("create model first\n");

	return result;
}
