#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"

class CGUI ;

class CGUIPreProc
{
	public:
		CGUIPreProc(CGUI*);
		~CGUIPreProc();

		bool set_preproc(char* param);
		inline CPreProc * get_preproc() { return preproc ; }
		bool load(char* param);
		bool save(char* param);
	protected:
		CGUI* gui ;
		CPreProc * preproc;
};
#endif
