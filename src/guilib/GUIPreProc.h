#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"

class CGUI ;

class CGUIPreProc
{
	public:
		CGUIPreProc(CGUI*);
		~CGUIPreProc();

		bool add_preproc(CHAR* param);
		bool del_preproc(CHAR* param);
		inline CPreProc** get_preprocs(INT &num) { num=num_preprocs; return preprocs; }
		bool load(CHAR* param);
		bool save(CHAR* param);

	protected:
		bool add_preproc(CPreProc* preproc);
		CGUI* gui ;
		INT num_preprocs;
		CPreProc** preprocs;
};
#endif
