#include "guilib/GUIKernel.h"
#include "guilib/GUIPluginEstimate.h"
#include "kernel/Kernel.h"
#include "kernel/RealKernel.h"
#include "kernel/ShortKernel.h"
#include "kernel/CharKernel.h"
#include "kernel/ByteKernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/LinearByteKernel.h"
#include "kernel/LinearCharKernel.h"
#include "kernel/LinearWordKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/FixedDegreeCharKernel.h"
#include "kernel/LocalityImprovedCharKernel.h"
#include "kernel/SimpleLocalityImprovedCharKernel.h"
#include "kernel/PolyKernel.h"
#include "kernel/CharPolyKernel.h"
#include "kernel/PolyMatchWordKernel.h"
#include "kernel/WordMatchKernel.h"
#include "kernel/CommWordKernel.h"
#include "kernel/HistogramWordKernel.h"
#include "kernel/GaussianKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SparseGaussianKernel.h"
#include "kernel/SparseRealKernel.h"
#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/ByteFeatures.h"
#include "features/ShortFeatures.h"
#include "features/RealFeatures.h"
#include "features/SparseRealFeatures.h"
#include "features/Features.h"
#include "lib/io.h"
#include "gui/GUI.h"

#include <string.h>

CGUIKernel::CGUIKernel(CGUI * gui_): gui(gui_)
{
	kernel=NULL;
	initialized=false;
}

CGUIKernel::~CGUIKernel()
{
	delete kernel;
}

CKernel* CGUIKernel::get_kernel()
{
	return kernel;
}

bool CGUIKernel::set_kernel(CHAR* param)
{
	INT size=100;
	CHAR kern_type[1024]="";
	CHAR data_type[1024]="";
	param=CIO::skip_spaces(param);
	
	if (sscanf(param, "%s %s %d", kern_type, data_type, &size) >= 2)
	{
		if (strcmp(kern_type,"LINEAR")==0)
		{
			if (strcmp(data_type,"BYTE")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete kernel;
				kernel=new CLinearByteKernel(size);
				if (kernel)
				{
					CIO::message("LinearByteKernel created\n");
					return true;
				}
			}
			else if (strcmp(data_type,"WORD")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete kernel;
				kernel=new CLinearWordKernel(size);
				if (kernel)
				{
					CIO::message("LinearWordKernel created\n");
					return true;
				}
			}
			else if (strcmp(data_type,"CHAR")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete kernel;
				kernel=new CLinearCharKernel(size);
				if (kernel)
				{
					CIO::message("LinearCharKernel created\n");
					return true;
				}
			}
			else if (strcmp(data_type,"REAL")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete kernel;
				kernel=new CLinearKernel(size);
				return true;
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete kernel;
				kernel=new CSparseLinearKernel(size);
				return true;
			}
		}
		else if (strcmp(kern_type,"HISTOGRAM")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				if (kernel)
				  {
				    CIO::message("destroying old kernel\n") ;
				    delete kernel;
				  } ;

				CIO::message("getting estimator\n") ;
				CPluginEstimate* estimator=gui->guipluginestimate.get_estimator();

				if (estimator)
					kernel=new CHistogramWordKernel(size, estimator);
				else
					CIO::message("no estimator set\n");

				if (kernel)
				{
					CIO::message("HistogramKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"POLYMATCH")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				INT inhomogene=0;
				INT degree=2;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &degree, &inhomogene);
				delete kernel;
				kernel=new CPolyMatchWordKernel(size, degree, inhomogene==1);

				if (kernel)
				{
					CIO::message("PolyMatchWordKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"MATCH")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				delete kernel;
				INT d=3;
				sscanf(param, "%s %s %d %d", kern_type, data_type, &size, &d);
				kernel=new CWordMatchKernel(size, d);

				if (kernel)
				{
					CIO::message("CommWordKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"COMM")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				delete kernel;
				kernel=new CCommWordKernel(size);

				if (kernel)
				{
					CIO::message("CommWordKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"FIXEDDEGREE")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT d=3;

				sscanf(param, "%s %s %d %d", kern_type, data_type, &size, &d);
				delete kernel;
				kernel=new CFixedDegreeCharKernel(size, d);

				if (kernel)
				{
					CIO::message("FixedDegreeCharKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"WEIGHTEDDEGREE")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT d=3;
				INT max_mismatch = 0 ;
				INT i=0;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &d, &max_mismatch);
				REAL* weights=new REAL[d*(1+max_mismatch)];
				REAL sum=0;

				for (i=0; i<d; i++)
				{
					weights[i]=d-i;
					sum+=weights[i];
				}
				for (i=0; i<d; i++)
					weights[i]/=sum;
				
				for (i=0; i<d; i++)
				{
					for (INT j=1; j<=max_mismatch; j++)
					{
						if (j<i+1)
						{
							INT nk=math.nchoosek(i+1, j) ;
							weights[i+j*d]=weights[i]/(nk*pow(3,j)) ;
						}
						else
							weights[i+j*d]= 0;
						
					} ;
				} ;
				/*for (i=0; i<d; i++)
				{
					for (INT j=0; j<=max_mismatch; j++)
					{
						CIO::message("%1.3f  ", weights[i+j*d]) ;
					}
					CIO::message("\n") ;
					} ;*/
				
				delete kernel;
				kernel=new CWeightedDegreeCharKernel(size, weights, d, max_mismatch);

				if (kernel)
				{
					CIO::message("WeightedDegreeCharKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"SLIK")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT l=3;
				INT d1=3;
				INT d2=1;
				sscanf(param, "%s %s %d %d %d %d", kern_type, data_type, &size, &l, &d1, &d2);
				delete kernel;
				kernel=new CSimpleLocalityImprovedCharKernel(size, l, d1, d2);
				if (kernel)
				{
					CIO::message("SimpleLocalityImprovedCharKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"LIK")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT l=3;
				INT d1=3;
				INT d2=1;
				sscanf(param, "%s %s %d %d %d %d", kern_type, data_type, &size, &l, &d1, &d2);
				delete kernel;
				kernel=new CLocalityImprovedCharKernel(size, l, d1, d2);
				if (kernel)
				{
					CIO::message("LocalityImprovedCharKernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"POLY")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				INT inhomogene=0;
				INT degree=2;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &degree, &inhomogene);
				delete kernel;
				kernel=new CPolyKernel(size, degree, inhomogene==1);

				if (kernel)
				{
					CIO::message("Polynomial Kernel created\n");
					return true;
				}
			}
			else if (strcmp(data_type,"CHAR")==0)
			{
				INT inhomogene=0;
				INT degree=2;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &degree, &inhomogene);
				delete kernel;
				kernel=new CCharPolyKernel(size, degree, inhomogene==1);

				if (kernel)
				{
					CIO::message("CharPolynomial Kernel created\n");
					return true;
				}
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				INT inhomogene=0;
				INT degree=2;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &degree, &inhomogene, &size);
				delete kernel;
				kernel=new CSparsePolyKernel(size, degree, inhomogene==1);

				if (kernel)
				{
					CIO::message("Sparse Polynomial Kernel created\n");
					return true;
				}
			}
		}
		else if (strcmp(kern_type,"GAUSSIAN")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				double width=1;

				sscanf(param, "%s %s %lf %d", kern_type, data_type, &width, &size);
				delete kernel;
				kernel=new CGaussianKernel(size, width);
				if (kernel)
				{
					CIO::message("Gaussian Kernel created\n");
					return true;
				}
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				double width=1;

				sscanf(param, "%s %s %lf %d", kern_type, data_type, &width, &size);
				delete kernel;
				kernel=new CSparseGaussianKernel(size, width);
				if (kernel)
				{
					CIO::message("Sparse Gaussian Kernel created\n");
					return true;
				}
			}
		}
		else 
			CIO::not_implemented();
	}
	else 
		CIO::message("see help for params!\n");

	CIO::not_implemented();
	return false;
}

bool CGUIKernel::load_kernel_init(CHAR* param)
{
	bool result=false;
	CHAR filename[1024]="";

	if (kernel)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			FILE* file=fopen(filename, "r");
			if ((!file) || (!kernel->load_init(file)))
				CIO::message("reading from file %s failed!\n", filename);
			else
			{
				CIO::message("successfully read kernel init data from \"%s\" !\n", filename);
				initialized=true;
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			CIO::message("see help for params\n");
	}
	else
		CIO::message("no kernel set!\n");
	return result;
}

bool CGUIKernel::save_kernel_init(CHAR* param)
{
	bool result=false;
	CHAR filename[1024]="";

	if (kernel)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			FILE* file=fopen(filename, "w");
			if (!file)
				CIO::message("fname: %s\n", filename);
			if ((!file) || (!kernel->save_init(file)))
				CIO::message("writing to file %s failed!\n", filename);
			else
			{
				CIO::message("successfully written kernel init data into \"%s\" !\n", filename);
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			CIO::message("see help for params\n");
	}
	else
		CIO::message("no kernel set!\n");
	return result;
}

bool CGUIKernel::init_kernel(CHAR* param)
{
	CHAR target[1024]="";
	bool do_init=false;

	if (!kernel)
	  {
	    CIO::message("no kernel available\n") ;
	    return false ;
	  } ;

	if ((sscanf(param, "%s", target))==1)
	{
	  if (!strncmp(target, "TRAIN", 5))
	    {
	      do_init=true;
	      if (gui->guifeatures.get_train_features())
		{
		  if ( (kernel->get_feature_class() == gui->guifeatures.get_train_features()->get_feature_class()) &&
		       (kernel->get_feature_type() == gui->guifeatures.get_train_features()->get_feature_type()))
		    {
		      kernel->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_train_features(), do_init);
		      initialized=true;
		    }
		  else
		    {
		      CIO::message("kernel can not process this feature type\n");
		      return false ;
		    }
		}
	      else
		CIO::message("assign train features first\n");
	    }
	  else if (!strncmp(target, "TEST", 5))
	    {
	      if (gui->guifeatures.get_train_features() && gui->guifeatures.get_test_features())
		{
		  if	(((kernel->get_feature_class() == gui->guifeatures.get_train_features()->get_feature_class()) && 
			  (kernel->get_feature_class() == gui->guifeatures.get_test_features()->get_feature_class())) &&
			 ((kernel->get_feature_type() == gui->guifeatures.get_train_features()->get_feature_type()) && 
			  (kernel->get_feature_type() == gui->guifeatures.get_test_features()->get_feature_type())) )
		    {
		      if (!initialized)
			{
			  CIO::message("kernel not initialized for training examples\n") ;
			  return false ;
			}
		      else
			{
			  CIO::message("initialising kernel with TEST DATA, train: %d test %d\n",gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features() );
			  // lhs -> always train_features; rhs -> always test_features
			  kernel->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features(), do_init);
			} ;
		    }
		  else
		    {
		      CIO::message("kernel can not process this feature type\n");
		      return false ;
		    }
		}
	      else
		CIO::message("assign train and test features first\n");
	      
	    }
	  else
	    CIO::not_implemented();
	}
	else 
	  {
	    CIO::message("see help for params\n");
	    return false;
	  }
	
	return true;
}

bool CGUIKernel::save_kernel(CHAR* param)
{
	bool result=false;
	CHAR filename[1024]="";

	if (kernel && initialized)
	{
		if ((sscanf(param, "%s", filename))==1)
		{
			if (!kernel->save(filename))
				CIO::message("writing to file %s failed!\n", filename);
			else
			{
				CIO::message("successfully written kernel to \"%s\" !\n", filename);
				result=true;
			}
		}
		else
			CIO::message("see help for params\n");
	}
	else
		CIO::message("no kernel set / kernel not initialized!\n");
	return result;
}
