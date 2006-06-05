#include "guilib/GUITime.h"

#include <assert.h>

CGUITime::CGUITime(CGUI* g) : gui(g)
{
	time=new CTime();
	assert(time);
}

CGUITime::~CGUITime()
{
	delete time;
}

void CGUITime::start()
{
	time->start();
}

void CGUITime::stop()
{
	time->stop();
	time->time_diff_sec(true);
}
