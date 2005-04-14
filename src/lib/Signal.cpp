#include <stdlib.h>
#include <signal.h>
#include <string.h>

#include "lib/config.h"
#include "lib/io.h"
#include "lib/Signal.h"

struct sigaction CSignal::oldsigaction;

CSignal::CSignal()
{
	memset(&CSignal::oldsigaction, 0, sizeof(CSignal::oldsigaction));
	CIO::message(M_INFO, "initalizing signal handler\n");

	if (!set_handler())
		CIO::message(M_ERROR, "error initalizing signal handler\n");
}

CSignal::~CSignal()
{
	if (!unset_handler())
		CIO::message(M_ERROR, "error deinitalizing signal handler\n");
}

void CSignal::handler(int signal)
{
#ifdef HAVE_MATLAB
	unset_handler();
	CIO::message(M_MESSAGEONLY, "\n");
	CIO::message(M_ERROR, "gf stopped by SIGTERM\n");
#else
	CIO::message(M_MESSAGEONLY, "\n");
	CIO::message(M_ERROR, "gf stopped by SIGTERM\n");
	unset_handler();
	exit(0);
#endif
}

bool CSignal::set_handler()
{
	struct sigaction act;
	sigset_t st;

	sigemptyset(&st);

	act.sa_restorer = NULL; //just in case remove
	act.sa_sigaction=NULL; //just in case
	act.sa_handler=CSignal::handler;
	act.sa_mask = st;
	act.sa_flags = 0;

	if (!sigaction(SIGTERM, &act, &oldsigaction))
		return true;
	else
		return false;
}

bool CSignal::unset_handler()
{
	if (!sigaction(SIGTERM, &oldsigaction, NULL))
		return true;
	else
		return false;
}
