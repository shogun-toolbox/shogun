#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"
#include "hmm/Observation.h"
#include "lib/io.h"
#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"

CGUIFeatures::CGUIFeatures(CGUI * gui_)
  : gui(gui_), train_features(NULL), test_features(NULL), train_obs(NULL), test_obs(NULL)
{
}

CGUIFeatures::~CGUIFeatures()
{
    delete train_features;
    delete test_features;
    delete train_obs;
    delete test_obs;
}
		
bool CGUIFeatures::preprocess(char* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	char target[1024];
	int force=0;

	if ((sscanf(param, "%s %d", target, &force))>=1)
	{
		if ( strcmp(target, "TRAIN")==0 || strcmp(target, "TEST")==0 )
		{
			CFeatures** f_ptr=NULL;

			if (strcmp(target,"TRAIN")==0)
			{
				f_ptr=&train_features;
			}
			else if (strcmp(target,"TEST")==0)
			{
				f_ptr=&test_features;
			}
			else
				CIO::message("see help for parameters\n");

			if (*f_ptr)
			{
				int num_preproc=0;
				CPreProc** preprocs;
				if ((preprocs=gui->guipreproc.get_preprocs(num_preproc))!=NULL)
				{
					if (train_features)
					{
						CFeatures* feat=train_features;
						for (int i=0; i<num_preproc; i++)
						{
							preprocs[i]->init(feat);
							feat->add_preproc(preprocs[i]);
							if (feat != *f_ptr)
								(*f_ptr)->add_preproc(preprocs[i]);
						}
						
						CIO::message("force: %d\n",force);
						(*f_ptr)->preproc_feature_matrix(force==1);
					}
					else
						CIO::message("initialization of preprocessor needs trainfeatures !\n");
				}
				else
					CIO::message("no preprocessor set!\n");
			}
			else
				CIO::message("features not correctly assigned!\n");
		}
		else
			CIO::message("observations not correctly assigned!\n");
	}
	else
		CIO::message("see help for parameters\n");

	return result;
}

bool CGUIFeatures::set_features(char* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	char target[1024];
	char type[1024];
	int comp_features=1;
	int size=100;
	int neglinear=0;
	int poslinear=0;

	if ((sscanf(param, "%s %s %d %d %d %d", type, target, &size, &comp_features, &neglinear, &poslinear))>=2)
	{
		if ( (strcmp(target, "TRAIN")==0 && gui->guiobs.get_obs("POSTRAIN") && gui->guiobs.get_obs("NEGTRAIN")) ||
				(strcmp(target, "TEST")==0 && gui->guiobs.get_obs("POSTEST") && gui->guiobs.get_obs("NEGTEST")))
		{
			CFeatures** f_ptr=NULL;
			CObservation** o_ptr=NULL;
			CObservation* pt=NULL;
			CObservation* nt=NULL;

			if (strcmp(target,"TRAIN")==0)
			{
				f_ptr=&train_features;
				o_ptr=&train_obs;
				pt=gui->guiobs.get_obs("POSTRAIN") ;
				nt=gui->guiobs.get_obs("NEGTRAIN") ;
			}
			else if (strcmp(target,"TEST")==0)
			{
				f_ptr=&test_features;
				o_ptr=&test_obs;
				pt=gui->guiobs.get_obs("POSTEST") ;
				nt=gui->guiobs.get_obs("NEGTEST") ;
			}
			else
				CIO::message("see help for parameters\n");

			if (strcmp(type,"TOP")==0)
			{
				CIO::message("setting TOP features\n");
				if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
				{

					//CObservation* old_obs_pos=gui->guihmm.get_pos()->get_observations();
					//CObservation* old_obs_neg=gui->guihmm.get_neg()->get_observations();

					delete (*o_ptr);
					*o_ptr=new CObservation(pt, nt);
					gui->guihmm.get_pos()->set_observations(*o_ptr);
					gui->guihmm.get_neg()->set_observations(*o_ptr);

					delete (*f_ptr);
					*f_ptr= new CTOPFeatures(size, gui->guihmm.get_pos(), gui->guihmm.get_neg(), neglinear, poslinear);		     


					//						gui->guihmm.get_pos()->set_observations(old_obs_pos);
					//						gui->guihmm.get_neg()->set_observations(old_obs_neg);

				}
				else
					CIO::message("HMMs not correctly assigned!\n");
			}
			else if (strcmp(type,"FK")==0)
			{
				CIO::message("setting FK features\n");
				if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
				{

					//CObservation* old_obs_pos=gui->guihmm.get_pos()->get_observations();
					//CObservation* old_obs_neg=gui->guihmm.get_neg()->get_observations();

					delete (*o_ptr);
					*o_ptr=new CObservation(pt, nt);
					gui->guihmm.get_pos()->set_observations(*o_ptr);
					gui->guihmm.get_neg()->set_observations(*o_ptr);

					delete (*f_ptr);
					*f_ptr= new CFKFeatures(size, gui->guihmm.get_pos(), gui->guihmm.get_neg());

					if (train_features)
						((CFKFeatures*) *f_ptr)->set_opt_a(((CFKFeatures*) train_features)->get_weight_a());
					else
					{
						delete (*f_ptr);
						(f_ptr)=NULL;
						CIO::message("no train features recognized.\n");
					}

					//						gui->guihmm.get_pos()->set_observations(old_obs_pos);
					//						gui->guihmm.get_neg()->set_observations(old_obs_neg);

				}
				else
					CIO::message("HMMs not correctly assigned!\n");
			}
			else
				CIO::not_implemented();

			if (comp_features)
				((CRealFeatures*) *f_ptr)->set_feature_matrix() ;
		}
		else
			CIO::message("observations not correctly assigned!\n");
	}
	else
		CIO::message("see help for parameters\n");

	return result;
}

bool CGUIFeatures::load(char* param)
{
    param=CIO::skip_spaces(param);
    char filename[1024];
    char target[1024];
    char type[1024];
    bool result=false;
	int comp_features=0;
	int size=100;

    if ((sscanf(param, "%s %s %s %d  %d", filename, type, target, &size, &comp_features))>=3)
    {
	CFeatures** f_ptr=NULL;

	if (strcmp(target,"TRAIN")==0)
	{
	    f_ptr=&train_features;
	}
	else if (strcmp(target,"TEST")==0)
	{
	    f_ptr=&test_features;
	}
	else
	{
	    CIO::message("see help for parameters\n");
	    return false;
	}

	delete (*f_ptr);
	*f_ptr=NULL;

	if (strcmp(type,"REAL")==0)
	{
	    CIO::message("opening file...\n");
	    *f_ptr=new CRealFileFeatures(size, filename);
	}
	else
	{
	    CIO::message("unknown type\n");
	    return false;
	}
	if (comp_features)
		((CRealFeatures*) *f_ptr)->set_feature_matrix() ;

    } else
	CIO::message("see help for params\n");

    return result;
}

bool CGUIFeatures::save(char* param)
{
    bool result=false;
    param=CIO::skip_spaces(param);
    char filename[1024];
    char target[1024];

    if ((sscanf(param, "%s %s", filename, target))==2)
    {

	CFeatures** f_ptr=NULL;

	if (strcmp(target,"TRAIN")==0)
	{
	    f_ptr=&train_features;
	}
	else if (strcmp(target,"TEST")==0)
	{
	    f_ptr=&test_features;
	}
	else
	{
	    CIO::message("see help for parameters\n");
	    return false;
	}

	if (*f_ptr)
	{
	    if ((filename) || (!(*f_ptr)->save(filename)))
		CIO::message("writing to file %s failed!\n", filename);
	    else
	    {
		CIO::message("successfully written features into \"%s\" !\n", filename);
		result=true;
	    }

	} else
	    CIO::message("set features first\n");

    } else
	CIO::message("see help for params\n");

    return result;
}
