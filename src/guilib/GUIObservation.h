#ifndef __GUIOBSERVATION__H
#define __GUIOBSERVATION__H

#include "hmm/Observation.h"

class CGUI ;

class CGUIObservation
{
public:
	CGUIObservation(CGUI *);
	~CGUIObservation();
	bool load_observations(char* param);
	CObservation* get_obs(char* param);

protected:
	CObservation* pos_train_obs;
	CObservation* neg_train_obs;
	CObservation* pos_test_obs;
	CObservation* neg_test_obs;
	CObservation* test_obs;

	E_OBS_ALPHABET alphabet;

 protected:
	CGUI* gui ;
};
#endif
