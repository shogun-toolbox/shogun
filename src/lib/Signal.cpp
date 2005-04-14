#include <stdlib.h>
#include <signal.h>

#include "lib/config.h"
#include "lib/io.h"
#include "lib/Signal.h"

CSignal::CSignal()
{
	set_handler();
}

CSignal::~CSignal()
{
	unset_handler();
}

void CSignal::handler(int signal)
{
#ifdef HAVE_MATLAB
	unset_handler();
	CIO::message(M_ERROR, "gf stopped by SIGTERM\n");
#else
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
	act.sa_flags = SA_NODEFER;

	return (sigaction(SIGTERM, &act, NULL) == 0);
}

bool CSignal::unset_handler()
{
	struct sigaction act;
	sigset_t st;

	sigemptyset(&st);

	act.sa_restorer = NULL; //just in case remove
	act.sa_sigaction=NULL; //just in case
	act.sa_handler=SIG_DFL;
	act.sa_mask = st;
	act.sa_flags = SA_NODEFER;

	return (sigaction(SIGTERM, &act, NULL) == 0);
}
