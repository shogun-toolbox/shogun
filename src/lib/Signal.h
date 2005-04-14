#ifndef __SIGNAL__H_
#define __SIGNAL__H_
class CSignal
{
public:
	CSignal();
	~CSignal();

	static void handler(int);

	static bool set_handler();
	static bool unset_handler();
};
#endif
