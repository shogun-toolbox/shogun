#include "guilib/GUISVM.h"
#include "gui/GUI.h"
#include "lib/io.h"
#include "features/RealFileFeatures.h"
#include "features/Labels.h"

#include "classifier/svm/SVM_light.h"
#include "classifier/svm_cplex/SVM_cplex.h"
#include "classifier/svm/LibSVM.h"

#ifdef USE_SVMMPI
#include "classifier/svm_mpi/mpi_oneclass.h"
#endif

#include <assert.h>

CGUISVM::CGUISVM(CGUI * gui_)
  : gui(gui_)
{
	use_mkl = true;
	svm=NULL;
	C1=-1;
	C2=-1;
	weight_epsilon=-1;
}

CGUISVM::~CGUISVM()
{
	delete svm;
}

bool CGUISVM::new_svm(CHAR* param)
{
	param=CIO::skip_spaces(param);

	if (strcmp(param,"LIGHT")==0)
	{
		delete svm;
		svm= new CSVMLight();
		CIO::message(M_INFO, "created SVMLight object\n") ;
	}
	else if (strcmp(param,"LIBSVM")==0)
	{
		delete svm;
		svm= new CLibSVM();
		CIO::message(M_INFO, "created SVMlibsvm object\n") ;
	}
	else if (strcmp(param,"CPLEX")==0)
	{
#ifdef SVMCPLEX
		delete svm;
		svm= new CSVMCplex();
		CIO::message(M_INFO, "created SVMCplex object\n") ;
#else
		CIO::message(M_ERROR, "CPLEX SVM disabled\n") ;
#endif
	}
	else if (strcmp(param,"MPI")==0)
	{
#ifdef USE_SVMMPI
		delete svm;
		svm= new CSVMMPI();
		CIO::message(M_INFO, "created SVMMPI object\n") ;
#else
		CIO::message(M_ERROR, "MPI SVM disabled\n") ;
#endif
	}
	else if (strcmp(param,"ONECLASS")==0)
	{
#ifdef USE_SVMMPI
		delete svm;
		svm= new COneClassMPI();
		CIO::message(M_INFO, "created OneClassMPI object\n") ;
#else
		CIO::message(M_ERROR, "MPI OneClass SVM disabled\n") ;
#endif
	}
	else
		return false;

	return (svm!=NULL);
}

bool CGUISVM::train(CHAR* param)
{
	param=CIO::skip_spaces(param);

	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CKernel* kernel=gui->guikernel.get_kernel();

	if (!svm)
	{
		CIO::message(M_ERROR, "no svm available\n") ;
		return false ;
	}

	if (!kernel)
	{
		CIO::message(M_ERROR, "no kernel available\n");
		return false ;
	}

	if (!trainlabels)
	{
		CIO::message(M_ERROR, "no trainlabels available\n");
		return false ;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return 0;
	}

	CIO::message(M_INFO, "starting svm training on %ld vectors using C1=%lf C2=%lf\n", trainlabels->get_num_labels(), C1, C2) ;

	svm->set_weight_epsilon(weight_epsilon);
	svm->set_C(C1, C2);
	svm->use_kerncombination(use_mkl);
	((CKernelMachine*) svm)->set_labels(trainlabels);
	((CKernelMachine*) svm)->set_kernel(kernel);
	return svm->train();
}

bool CGUISVM::test(CHAR* param)
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
			CIO::message(M_ERROR, "could not open %s\n",outputname);
			return false;
		}

		if (numargs==2) 
		{
			rocfile=fopen(rocfname, "w");

			if (!rocfile)
			{
				CIO::message(M_ERROR, "could not open %s\n",rocfname);
				return false;
			}
		}
	}

	CLabels* testlabels=gui->guilabels.get_test_labels();
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();
	CIO::message(M_DEBUG, "I:training: %ld examples each %ld features\n", ((CRealFeatures*) trainfeatures)->get_num_vectors(), ((CRealFeatures*) trainfeatures)->get_num_features());
	CIO::message(M_DEBUG, "I:testing: %ld examples each %ld features\n", ((CRealFeatures*) testfeatures)->get_num_vectors(), ((CRealFeatures*) testfeatures)->get_num_features());

	if (!svm)
	{
		CIO::message(M_ERROR, "no svm available") ;
		return false ;
	}
	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no training features available") ;
		return false ;
	}

	if (!testfeatures)
	{
		CIO::message(M_ERROR, "no test features available") ;
		return false ;
	}

	if (!testlabels)
	{
		CIO::message(M_ERROR, "no test labels available") ;
		return false ;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return 0;
	}

	CIO::message(M_INFO, "starting svm testing\n") ;
	((CKernelMachine*) svm)->set_labels(testlabels);
	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;

	if ( (gui->guikernel.get_kernel()->is_optimizable()) && (gui->guikernel.get_kernel()->get_is_initialized()))
		CIO::message(M_DEBUG, "using kernel optimization\n");

	REAL* output= svm->test();

	INT len=0;
	INT total=	testfeatures->get_num_vectors();
	INT* label= testlabels->get_int_labels(len);

	assert(label);
	CIO::message(M_DEBUG, "len:%d total:%d\n", len, total);
	assert(len==total);

	gui->guimath.evaluate_results(output, label, total, outputfile, rocfile);

	if (rocfile)
		fclose(rocfile);
	if ((outputfile) && (outputfile!=stdout))
		fclose(outputfile);

	delete[] output;
	delete[] label;
	return true;
}

bool CGUISVM::load(CHAR* param)
{
    bool result=false;
    param=CIO::skip_spaces(param);
    CHAR filename[1024];
    CHAR type[1024];

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
		    CIO::message(M_ERROR, "svm creation/loading failed\n");

		fclose(model_file);
	    }
	    else
		CIO::message(M_ERROR, "opening file %s failed\n", filename);

	    return result;
	}
	else
	    CIO::message(M_ERROR, "type of svm unknown\n");
    }
    else
	CIO::message(M_ERROR, "see help for parameters\n");
	return false;
}

bool CGUISVM::save(CHAR* param)
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
	CIO::message(M_ERROR, "create svm first\n");

    return result;
}

bool CGUISVM::set_weight_epsilon(CHAR* param)
{
	param=CIO::skip_spaces(param);

	sscanf(param, "%le", &weight_epsilon) ;

	if (weight_epsilon<0)
		weight_epsilon=1e-4;

	CIO::message(M_INFO, "Set to weight_epsilon=%f\n", weight_epsilon);
	return true ;  
}

bool CGUISVM::set_C(CHAR* param)
{
	param=CIO::skip_spaces(param);

	C1=-1;
	C2=-1;

	sscanf(param, "%le %le", &C1, &C2) ;

	if (C1<0)
		C1=1.0;
	if (C2<0)
		C2=C1;

	CIO::message(M_INFO, "Set to C1=%f C2=%f\n", C1, C2) ;
	return true ;  
}

bool CGUISVM::use_kerncombination(CHAR* param)
{
	param=CIO::skip_spaces(param);

	int mkl=1;
	sscanf(param, "%d", &mkl) ;

	use_mkl = (mkl==1);

	if (mkl)
		CIO::message(M_INFO, "Enabling MKL kern combination\n") ;
	else
		CIO::message(M_INFO, "Disabling MKL kern combination\n") ;

	return true ;  
}

CLabels* CGUISVM::classify(CLabels* output)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!svm)
	{
		CIO::message(M_ERROR, "no svm available\n") ;
		return NULL;
	}
	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no training features available\n") ;
		return NULL;
	}

	if (!testfeatures)
	{
		CIO::message(M_ERROR, "no test features available\n") ;
		return NULL;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return NULL;
	}
	  
	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;

	if ((gui->guikernel.get_kernel()->is_optimizable()) && (gui->guikernel.get_kernel()->get_is_initialized()))
		CIO::message(M_DEBUG, "using kernel optimization\n");

	CIO::message(M_INFO, "starting svm testing\n") ;
	return svm->classify(output);
}

bool CGUISVM::classify_example(INT idx, REAL &result)
{
	CFeatures* trainfeatures=gui->guifeatures.get_train_features();
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!svm)
	{
		CIO::message(M_ERROR, "no svm available\n") ;
		return false;
	}
	if (!trainfeatures)
	{
		CIO::message(M_ERROR, "no training features available\n") ;
		return false;
	}

	if (!testfeatures)
	{
		CIO::message(M_ERROR, "no test features available\n") ;
		return false;
	}

	if (!gui->guikernel.is_initialized())
	{
		CIO::message(M_ERROR, "kernel not initialized\n") ;
		return false;
	}

	((CKernelMachine*) svm)->set_kernel(gui->guikernel.get_kernel()) ;

	result=svm->classify_example(idx);
	return true ;
}
