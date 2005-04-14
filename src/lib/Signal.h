#ifndef __SIGNAL__H_
#define __SIGNAL__H_

#include <signal.h>

class CSignal
{
public:
	CSignal();
	~CSignal();

	static void handler(int);

	static bool set_handler();
	static bool unset_handler();
protected:
	static struct sigaction oldsigaction;
};
#endif
