#ifndef __GUITIME__H_
#define __GUITIME__H_

#include "lib/Time.h"

class CGUI;

class CGUITime
{
	public:
		CGUITime(CGUI *);
		~CGUITime();

		void start();
		void stop();

	protected:
		CGUI* gui;
		CTime* time;
};
#endif
