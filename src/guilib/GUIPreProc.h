#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"

class CGUI ;

class CGUIPreProc
{
public:
	CGUIPreProc(CGUI*);
	~CGUIPreProc();
 protected:
	CGUI* gui ;
};
#endif
