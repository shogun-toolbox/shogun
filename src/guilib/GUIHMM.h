#ifndef __GUIHMM__H
#define __GUIHMM__H

#include "hmm/Observation.h"
#include "hmm/HMM.h"

class CGUIHMM
{
	CGUIHMM();
	~CGUIHMM();

	CHMM* postrain;
	CHMM* negtrain;
	CHMM* postest;
	CHMM* negtest;
	CHMM* test;
	CHMM* train;
};
#endif
