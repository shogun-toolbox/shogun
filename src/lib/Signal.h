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
	static void clear();
	static inline bool cancel_computations() { return cancel_computation; }
protected:
	static struct sigaction oldsigaction;
	static bool active;
	static bool cancel_computation;
};
#endif
