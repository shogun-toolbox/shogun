#include "guilib/GUIHMM.h"

//#include <stdlib.h>
//#include <string.h>
#include "gui/GUI.h"

CGUIHMM::CGUIHMM(CGUI * gui_): gui(gui_)
{
	working=NULL;
	working_estimate=NULL;

	pos=NULL;
	neg=NULL;
	test=NULL;

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
	  if (working_estimate)
	    delete working_estimate;
	  working=new CHMM(n,m,order,NULL,PSEUDO);
	  working_estimate=new CHMM(n,m,order,NULL,PSEUDO);
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
	char templname[30]="/tmp/bw_model_XXXXXX" ;
	mkstemp(templname);
	char templname_best[40] ;
	sprintf(templname_best, "%s_best", templname) ;
	double prob_max=-CMath::INFTY ;
	iteration_count=ITERATIONS ;
	if (working) 
	{
		working_estimate=new CHMM(working);

		if (working->get_observations() && working_estimate->get_observations())
		{
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
				FILE* file=fopen(templname_best, "w");
				printf("\nsaving best model with filename %s ... ", templname_best) ;
				working->save_model(file) ;
				fclose(file) ;
				printf("done.") ;
			} 
			else
			{
				FILE* file=fopen(templname, "w");
				printf("\nsaving model with filename %s ... ", templname) ;
				working->save_model(file) ;
				fclose(file) ;
				printf("done.") ;
			} ;
			}
		}
		else
			printf("assign observation first\n");
	}
	else
	   CIO::message("create model first\n");

	return false;
}

bool CGUIHMM::linear_train(char* param)
{
	return false;
}

bool CGUIHMM::one_class_test(char* param)
{
	return false;
}

bool CGUIHMM::test_hmm(char* param)
{
	return false;
}

bool CGUIHMM::append_model(char* param)
{
	return false;
}

bool CGUIHMM::add_states(char* param)
{
	return false;
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
				pos=working;
				gui->guifeatures.set_hmms(pos, neg) ;
			}
			else if (strcmp(target,"NEG")==0)
			{
				neg=working;
				gui->guifeatures.set_hmms(pos, neg) ;
			}
			else if (strcmp(target,"TEST")==0)
			{
				test=working;
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
      if (working && working_estimate)
	{
	  CObservation *obs=gui->guiobs.get_obs(target) ;
	  working->set_observations(obs);
	  working_estimate->set_observations(obs);

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

bool CGUIHMM::load_hmm(char* param)
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

bool CGUIHMM::save_hmm(char* param)
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
