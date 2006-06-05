#include <assert.h>

#include "guilib/GUIKNN.h"
#include "lib/io.h"
#include "gui/GUI.h"

CGUIKNN::CGUIKNN(CGUI* g) : gui(g), knn(NULL), k(0)
{
}

CGUIKNN::~CGUIKNN()
{
}

bool CGUIKNN::new_knn(CHAR* param)
{
	knn=new CKNN(); 
	return true;
}

bool CGUIKNN::train(CHAR* param)
{
	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CKernel* kernel=gui->guikernel.get_kernel();

	bool result=false;

	if (trainlabels)
	{
		if (kernel)
		{
			param=CIO::skip_spaces(param);
			k=3;
			sscanf(param, "%d", &k);

			if (knn)
			{
				knn->set_labels(trainlabels);
				knn->set_kernel(kernel);
				knn->set_k(k);
				result=knn->train();
			}
			else
				CIO::message("no knn classifier available\n");
		}
		else
			CIO::message("no kernel available\n") ;
	}
	else
		CIO::message("no labels available\n") ;

	return result;
}

bool CGUIKNN::test(CHAR* param)
{
	CHAR outputname[1024];
	CHAR rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	INT numargs=-1;

	param=CIO::skip_spaces(param);

	numargs=sscanf(param, "%s %s", outputname, rocfname);

	if (numargs>=1)
	{
		outputfile=fopen(outputname, "w");

		if (!outputfile)
		{
			CIO::message(stderr,"ERROR: could not open %s\n",outputname);
			return false;
		}

		if (numargs==2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(stderr,"ERROR: could not open %s\n",rocfname);
				return false;
			}
		}
	}

	CLabels* testlabels=gui->guilabels.get_test_labels();
	CKernel* kernel=gui->guikernel.get_kernel();

	if (!knn)
	{
		CIO::message("no knn classifier available\n") ;
		return false ;
	}

	if (!kernel)
	{
		CIO::message("no kernel available\n") ;
		return false ;
	}

	if (!testlabels)
	{
		CIO::message("no test labels available\n") ;
		return false ;
	}

	knn->set_labels(testlabels);
	knn->set_kernel(kernel);

	CIO::message("starting knn classifier testing\n") ;
	REAL* output=knn->test();

	INT len=0;
	INT* label= testlabels->get_int_labels(len);
	assert(label);

	gui->guimath.evaluate_results(output, label, len, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true; 
}

bool CGUIKNN::load(CHAR* param)
{
  bool result=false;
  return result;
}

bool CGUIKNN::save(CHAR* param)
{
  bool result=false;
  return result;
}
