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
	bool set_alphabet(char* param);
	bool set_max_dim(char* param);
	int get_alphabet_size();
	
	inline char* get_test_name()
	{
		return test_name;
	}
	
	inline char* get_neg_test_name()
	{
		return neg_test_name;
	}

	inline char* get_pos_test_name()
	{
		return pos_test_name;
	}

protected:
	CObservation* pos_train_obs;
	CObservation* neg_train_obs;
	CObservation* pos_test_obs;
	CObservation* neg_test_obs;
	CObservation* test_obs;

	char* neg_test_name;
	char* pos_test_name;
	char* test_name;

	E_OBS_ALPHABET alphabet;

 protected:
	CGUI* gui ;
};
#endif
