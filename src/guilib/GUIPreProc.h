#ifndef __GUIPREPROC_H__
#define __GUIPREPROC_H__

#include "preproc/PreProc.h"

class CGUI ;

class CGUIPreProc
{
	public:
		CGUIPreProc(CGUI*);
		~CGUIPreProc();

		bool add_preproc(char* param);
		bool del_preproc(char* param);
		inline CPreProc** get_preprocs(int &num) { num=num_preprocs; return preprocs; }
		bool load(char* param);
		bool save(char* param);

	protected:
		bool add_preproc(CPreProc* preproc);
		CGUI* gui ;
		int num_preprocs;
		CPreProc** preprocs;
};
#endif
