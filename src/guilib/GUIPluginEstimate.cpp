#include <assert.h>

#include "guilib/GUIPluginEstimate.h"
#include "features/WordFeatures.h"
#include "lib/io.h"
#include "gui/GUI.h"

CGUIPluginEstimate::CGUIPluginEstimate(CGUI* g) : gui(g), estimator(NULL),pos_pseudo(1e-10), neg_pseudo(1e-10)
{
}

CGUIPluginEstimate::~CGUIPluginEstimate()
{
	delete estimator;
}

bool CGUIPluginEstimate::new_estimator(CHAR* param)
{
	delete estimator;
	estimator=new CPluginEstimate();
    return false;
}

bool CGUIPluginEstimate::train(CHAR* param)
{
	CLabels* trainlabels=gui->guilabels.get_train_labels();
	CWordFeatures* trainfeatures=(CWordFeatures*) gui->guifeatures.get_train_features();

	bool result=false;

	if (trainlabels)
	{
		if (trainfeatures)
		{
			assert(trainfeatures->get_feature_type()==F_WORD);

			param=CIO::skip_spaces(param);
			sscanf(param, "%le %le", &pos_pseudo, &neg_pseudo);

			if (estimator)
				result=estimator->train(trainfeatures, trainlabels, pos_pseudo, neg_pseudo);
			else
				CIO::message("no estimator available\n");
		}
		else
			CIO::message("no features available\n") ;
	}
	else
		CIO::message("no labels available\n") ;

	return result;
}

bool CGUIPluginEstimate::test(CHAR* param)
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
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!estimator)
	{
		CIO::message("no estimator available\n") ;
		return false ;
	}

	if (!estimator->check_models())
	{
		CIO::message("no models assigned\n") ;
		return false ;
	}

	if (!testfeatures || testfeatures->get_feature_class()!=C_SIMPLE || testfeatures->get_feature_type()!=F_WORD)
	{
		CIO::message("no test features of type WORD available\n") ;
		return false ;
	}

	if (!testlabels)
	{
		CIO::message("no test labels available\n") ;
		return false ;
	}

	CIO::message("starting estimator testing\n") ;
	estimator->set_testfeatures((CWordFeatures*) testfeatures);
	REAL* output=estimator->test();

	INT len=0;
	INT total=	testfeatures->get_num_vectors();
	INT* label= testlabels->get_int_labels(len);

	CIO::message("out !!! %ld %ld\n", total, len);
	assert(label);
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

bool CGUIPluginEstimate::load(CHAR* param)
{
  bool result=false;
  return result;
}

bool CGUIPluginEstimate::save(CHAR* param)
{
  bool result=false;
  return result;
}

CLabels* CGUIPluginEstimate::classify(CLabels* output)
{
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!estimator)
	{
		CIO::message("no estimator available") ;
		return 0;
	}

	if (!testfeatures)
	{
		CIO::message("no test features available") ;
		return 0;
	}

	estimator->set_testfeatures((CWordFeatures*) testfeatures);

	return estimator->classify(output);
}

REAL CGUIPluginEstimate::classify_example(INT idx)
{
	CFeatures* testfeatures=gui->guifeatures.get_test_features();

	if (!estimator)
	{
		CIO::message("no estimator available") ;
		return 0;
	}

	if (!testfeatures)
	{
		CIO::message("no test features available") ;
		return 0;
	}

	estimator->set_testfeatures((CWordFeatures*) testfeatures);

	return estimator->classify_example(idx);
}
