#include "guilib/GUISVM.h"
#include "gui/GUI.h"
#include "lib/io.h"
#include "features/RealFileFeatures.h"

#ifdef SVMMPI
#include "svm_mpi/mpi_oneclass.h"
#endif

#include <assert.h>

CGUISVM::CGUISVM(CGUI * gui_)
  : gui(gui_)
{
	svm=NULL;
}

CGUISVM::~CGUISVM()
{
	delete svm;
}

bool CGUISVM::new_svm(char* param)
{
  param=CIO::skip_spaces(param);
  
  if (strcmp(param,"LIGHT")==0)
    {
      delete svm;
      svm= new CSVMLight();
      CIO::message("created SVMLight object\n") ;
    }
  else if (strcmp(param,"CPLEX")==0)
    {
#ifdef SVMCPLEX
      delete svm;
      svm= new CSVMCplex();
      CIO::message("created SVMCplex object\n") ;
#else
      CIO::message("CPLEX SVM disabled\n") ;
#endif
    }
  else if (strcmp(param,"MPI")==0)
    {
#ifdef SVMMPI
      delete svm;
      svm= new CSVMMPI();
      CIO::message("created SVMMPI object\n") ;
#else
      CIO::message("MPI SVM disabled\n") ;
#endif
    }
  else if (strcmp(param,"ONECLASS")==0)
    {
#ifdef SVMMPI
      delete svm;
      svm= new COneClassMPI();
      CIO::message("created OneClassMPI object\n") ;
#else
      CIO::message("MPI OneClass SVM disabled\n") ;
#endif
    }
  else
    return false;
  
  return (svm!=NULL);
}

bool CGUISVM::train(char* param)
{
	param=CIO::skip_spaces(param);

	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CKernel* kernel=gui->guikernel.get_kernel();

	if (!svm)
	{
		CIO::message("no svm available\n") ;
		return false ;
	}

	if (!kernel)
	{
		CIO::message("no kernel available\n");
		return false ;
	}

	if (!trainlabels)
	{
		CIO::message("no trainlabels available\n");
		return false ;
	}

	CIO::message("starting svm training on %ld vectors using C=%lf\n", trainlabels->get_num_labels(), C) ;

	svm->set_C(C);
	((CKernelMachine*) svm)->set_labels(trainlabels);
	((CKernelMachine*) svm)->set_kernel(kernel);
	return svm->train();
}

bool CGUISVM::test(char* param)
{
	char outputname[1024];
	char rocfname[1024];
	FILE* outputfile=stdout;
	FILE* rocfile=NULL;
	int numargs=-1;

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
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	CIO::message("I:train features %ldx%ld\n", ((CRealFeatures*) trainfeatures)->get_num_vectors(), ((CRealFeatures*) trainfeatures)->get_num_features());
	CIO::message("I:test features %ldx%ld\n", ((CRealFeatures*) testfeatures)->get_num_vectors(), ((CRealFeatures*) testfeatures)->get_num_features());

	if (!svm)
	{
		CIO::message("no svm available") ;
		return false ;
	}
	if (!trainfeatures)
	{
		CIO::message("no training features available") ;
		return false ;
	}

	if (!testfeatures)
	{
		CIO::message("no test features available") ;
		return false ;
	}

	if (!testlabels)
	{
		CIO::message("no test labels available") ;
		return false ;
	}

	CIO::message("starting svm testing\n") ;
	((CKernelMachine*) svm)->set_labels(testlabels);
	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;

	REAL* output=NULL;
	///REAL* output= svm->test();

	long len=0;
	long total=	testfeatures->get_num_vectors();
	int* label=	testlabels->get_labels(len);

	CIO::message("out !!! %ld %ld\n", total, len);
	assert(label);
	assert(len==total);

	gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if (outputfile)
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true;
}

bool CGUISVM::load(char* param)
{
    bool result=false;
    param=CIO::skip_spaces(param);
    char filename[1024];
    char type[1024];

    if ((sscanf(param, "%s %s", filename, type))==2)
    {

	if (new_svm(type))
	{
	    FILE* model_file=fopen(filename, "r");

	    if (model_file)
	    {
		if (svm && svm->load(model_file))
		{
		    printf("file successfully read\n");
		    result=true;
		}
		else
		    CIO::message("svm creation/loading failed\n");

		fclose(model_file);
	    }
	    else
		CIO::message("opening file %s failed\n", filename);

	    return result;
	}
	else
	    CIO::message("type of svm unknown\n");
    }
    else
	CIO::message("see help for parameters\n");
	return false;
}

bool CGUISVM::save(char* param)
{
    bool result=false;
    param=CIO::skip_spaces(param);

    if (svm)
    {
	FILE* file=fopen(param, "w");

	if ((!file) ||	(!svm->save(file)))
	    printf("writing to file %s failed!\n", param);
	else
	{
	    printf("successfully written svm into \"%s\" !\n", param);
	    result=true;
	}

	if (file)
	    fclose(file);
    }
    else
	CIO::message("create svm first\n");

    return result;
}

bool CGUISVM::set_C(char* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &C) ;
	CIO::message("Set to C=%f\n", C) ;
	return true ;  
}
