#include "guilib/GUIFeatures.h"
#include "gui/GUI.h"
#include "hmm/Observation.h"
#include "lib/io.h"

CGUIFeatures::CGUIFeatures(CGUI * gui_)
  : gui(gui_), train_features(NULL), test_features(NULL)
{
}

CGUIFeatures::~CGUIFeatures()
{
}
		
bool CGUIFeatures::set_features(char* param)
{
	bool result=false;
	param=CIO::skip_spaces(param);
	char target[1024];
	char type[1024];

	if ((sscanf(param, "%s %s", type, target))==2)
	{
		if ( (strcmp(target, "TRAIN")==0 && gui->guiobs.get_obs("POSTRAIN") && gui->guiobs.get_obs("NEGTRAIN")) ||
			 (strcmp(target, "TEST")==0 && gui->guiobs.get_obs("POSTEST") && gui->guiobs.get_obs("NEGTEST")))
		{
			CFeatures** f_ptr=NULL;
			CObservation* pt=NULL;
			CObservation* nt=NULL;

			if (strcmp(target,"TRAIN")==0)
			{
				f_ptr=&train_features;
				pt=gui->guiobs.get_obs("POSTRAIN") ;
				nt=gui->guiobs.get_obs("NEGTRAIN") ;
			}
			else if (strcmp(target,"TEST")==0)
			{
				f_ptr=&test_features;
				pt=gui->guiobs.get_obs("POSTEST") ;
				nt=gui->guiobs.get_obs("NEGTEST") ;
			}
			else
				CIO::message("see help for parameters\n");

			if (&f_ptr)
			{
				if (strcmp(type,"TOP")==0)
				{
					if (gui->guihmm.get_pos() && gui->guihmm.get_neg())
					{

						CObservation* old_obs_pos=gui->guihmm.get_pos()->get_observations();
						CObservation* old_obs_neg=gui->guihmm.get_neg()->get_observations();

						CObservation* obs=new CObservation(pt, nt);
						gui->guihmm.get_pos()->set_observations(obs);
						gui->guihmm.get_neg()->set_observations(obs);

						delete (*f_ptr);
						*f_ptr= new CTOPFeatures(gui->guihmm.get_pos(), gui->guihmm.get_neg());
						((CTOPFeatures*) *f_ptr)->set_feature_matrix();

						gui->guihmm.get_pos()->set_observations(old_obs_pos);
						gui->guihmm.get_neg()->set_observations(old_obs_neg);

					}
					else
						CIO::message("HMMs not correctly assigned!\n");
				}
				else if (strcmp(type,"FK")==0)
				{
					CIO::not_implemented();
				}
				else
					CIO::not_implemented();
			}
		}
		else
			CIO::message("observations not correctly assigned!\n");
	}
	else
		CIO::message("see help for parameters\n");

	return result;
}
