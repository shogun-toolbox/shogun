#include "gui/GUI.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIPluginEstimate.h"
#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/RealKernel.h"
#include "kernel/ShortKernel.h"
#include "kernel/CharKernel.h"
#include "kernel/ByteKernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/LinearByteKernel.h"
#include "kernel/LinearCharKernel.h"
#include "kernel/LinearWordKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/WeightedDegreePositionCharKernel.h"
#include "kernel/FixedDegreeCharKernel.h"
#include "kernel/LocalityImprovedCharKernel.h"
#include "kernel/SimpleLocalityImprovedCharKernel.h"
#include "kernel/PolyKernel.h"
#include "kernel/CharPolyKernel.h"
#include "kernel/PolyMatchWordKernel.h"
#include "kernel/WordMatchKernel.h"
#include "kernel/CommWordKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/HistogramWordKernel.h"
#include "kernel/SalzbergWordKernel.h"
#include "kernel/GaussianKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SparseGaussianKernel.h"
#include "kernel/SparseNormSquaredKernel.h"
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
	CKernel* k=create_kernel(param);

	if (kernel && k)
		delete kernel;

	if (k)
	{
		kernel=k;
		return true;
	}
	else
	{
		CIO::message(M_ERROR, "kernel creation failed.\n");
		return false;
	}
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
				CIO::message(M_ERROR, "reading from file %s failed!\n", filename);
			else
			{
				CIO::message(M_INFO, "successfully read kernel init data from \"%s\" !\n", filename);
				initialized=true;
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			CIO::message(M_ERROR, "see help for params\n");
	}
	else
		CIO::message(M_ERROR, "no kernel set!\n");
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
				CIO::message(M_ERROR, "fname: %s\n", filename);
			if ((!file) || (!kernel->save_init(file)))
				CIO::message(M_ERROR, "writing to file %s failed!\n", filename);
			else
			{
				CIO::message(M_INFO, "successfully written kernel init data into \"%s\" !\n", filename);
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			CIO::message(M_ERROR, "see help for params\n");
	}
	else
		CIO::message(M_ERROR, "no kernel set!\n");
	return result;
}

bool CGUIKernel::init_kernel_tree(CHAR* param)
{
	if (gui->guisvm.get_svm()!=NULL)
	{
		CWeightedDegreeCharKernel *kernel_=
			(CWeightedDegreeCharKernel*)kernel ;
		
		if ((kernel->get_kernel_type() == K_WEIGHTEDDEGREE) && 
			(kernel_->get_max_mismatch()==0))
		{
			for(INT i=0; i<gui->guisvm.get_svm()->get_num_support_vectors(); i++)
			{
				if (i%1000==0)
					CIO::message(M_MESSAGEONLY, ".") ;
				kernel_->add_example_to_tree(gui->guisvm.get_svm()->get_support_vector(i), gui->guisvm.get_svm()->get_alpha(i)) ;
			}
		}
	}
	else
	{
		CIO::message(M_ERROR, "create SVM first\n");
		return false ;
	}
	return true ;
}

bool CGUIKernel::delete_kernel_tree(CHAR* param)
{
	CWeightedDegreeCharKernel *kernel_=
		(CWeightedDegreeCharKernel*)kernel ;
	
	if ((kernel->get_kernel_type() == K_WEIGHTEDDEGREE) && 
		(kernel_->get_max_mismatch()==0))
	{
		kernel_->delete_tree() ;
	} ;
	return true ;
}


bool CGUIKernel::init_kernel(CHAR* param)
{
	CHAR target[1024]="";
	bool do_init=false;

	if (!kernel)
	  {
	    CIO::message(M_ERROR, "no kernel available\n") ;
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
		      CIO::message(M_ERROR, "kernel can not process this feature type\n");
		      return false ;
		    }
		}
	      else
		CIO::message(M_ERROR, "assign train features first\n");
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
						CIO::message(M_ERROR, "kernel not initialized for training examples\n") ;
						return false ;
					}
					else
					{
						CIO::message(M_INFO, "initialising kernel with TEST DATA, train: %d test %d\n",gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features() );
						// lhs -> always train_features; rhs -> always test_features
						kernel->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features(), do_init);						
					} ;
				}
				else
				{
					CIO::message(M_ERROR, "kernel can not process this feature type\n");
					return false ;
				}
			}
			else
				CIO::message(M_ERROR, "assign train and test features first\n");
			
	    }
	  else
		  CIO::not_implemented();
	}
	else 
	{
	    CIO::message(M_ERROR, "see help for params\n");
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
				CIO::message(M_ERROR, "writing to file %s failed!\n", filename);
			else
			{
				CIO::message(M_INFO, "successfully written kernel to \"%s\" !\n", filename);
				result=true;
			}
		}
		else
			CIO::message(M_ERROR, "see help for params\n");
	}
	else
		CIO::message(M_ERROR, "no kernel set / kernel not initialized!\n");
	return result;
}

CKernel* CGUIKernel::create_kernel(CHAR* param)
{
	INT size=100;
	CHAR kern_type[1024]="";
	CHAR data_type[1024]="";
	param=CIO::skip_spaces(param);
	CKernel* k=NULL;

	//note the different args COMBINED <cachesize>
	if (sscanf(param, "%s %d", kern_type, &size) == 2)
	{
		if (strcmp(kern_type,"COMBINED")==0)
		{
			delete k;
			k= new CCombinedKernel(size);
			if (kernel)
				CIO::message(M_INFO, "CombinedKernel created\n");
			return k;
		}
	} 
	//compared with <KERNTYPE> <DATATYPE> <CACHESIZE>
	else if (sscanf(param, "%s %s %d", kern_type, data_type, &size) >= 2)
	{
		if (strcmp(kern_type,"LINEAR")==0)
		{
			if (strcmp(data_type,"BYTE")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CLinearByteKernel(size);
				if (k)
				{
					CIO::message(M_INFO, "LinearByteKernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"WORD")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CLinearWordKernel(size);
				if (k)
				{
					CIO::message(M_INFO, "LinearWordKernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"CHAR")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CLinearCharKernel(size);
				if (k)
				{
					CIO::message(M_INFO, "LinearCharKernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"REAL")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CLinearKernel(size);
				return k;
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CSparseLinearKernel(size);
				return k;
			}
		}
		else if (strcmp(kern_type,"HISTOGRAM")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				if (k)
				  {
				    CIO::message(M_INFO, "destroying old k\n") ;
				    delete k;
				  } ;

				CIO::message(M_INFO, "getting estimator\n") ;
				CPluginEstimate* estimator=gui->guipluginestimate.get_estimator();

				if (estimator)
					k=new CHistogramWordKernel(size, estimator);
				else
					CIO::message(M_ERROR, "no estimator set\n");

				if (k)
				{
					CIO::message(M_INFO, "HistogramKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"SALZBERG")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				if (k)
				{
				    CIO::message(M_INFO, "destroying old k\n") ;
				    delete k;
				  } ;

				CIO::message(M_INFO, "getting estimator\n") ;
				CPluginEstimate* estimator=gui->guipluginestimate.get_estimator();

				CIO::message(M_INFO, "getting labels\n") ;
				CLabels * train_labels = gui->guilabels.get_train_labels() ;
				if (!train_labels)
				{
					CIO::message(M_INFO, "assign train labels first!\n") ;
					return NULL ;
				} ;
				
				INT num_pos=0, num_neg=0;
				
				for (INT i=0; i<train_labels->get_num_labels(); i++)
				{
					if (train_labels->get_int_label(i)==1) num_pos++ ;
					if (train_labels->get_int_label(i)==-1) num_neg++ ;
				}				
				CIO::message(M_INFO, "priors: pos=%1.3f (%i)  neg=%1.3f (%i)\n", 
							 (REAL) num_pos/(num_pos+num_neg), num_pos,
							 (REAL) num_neg/(num_pos+num_neg), num_neg) ;
				
				if (estimator)
					k=new CSalzbergWordKernel(size, estimator);
				else
					CIO::message(M_ERROR, "no estimator set\n");
				
				((CSalzbergWordKernel*)k)->set_prior_probs((REAL)num_pos/(num_pos+num_neg), 
																(REAL)num_neg/(num_pos+num_neg)) ;
				
				if (k)
				{
					CIO::message(M_INFO, "SalzbergKernel created\n");
					return k;
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
				delete k;
				k=new CPolyMatchWordKernel(size, degree, inhomogene==1);

				if (k)
				{
					CIO::message(M_INFO, "PolyMatchWordKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"MATCH")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				delete k;
				INT d=3;
				sscanf(param, "%s %s %d %d", kern_type, data_type, &size, &d);
				k=new CWordMatchKernel(size, d);

				if (k)
				{
					CIO::message(M_INFO, "MatchKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"COMM")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				delete k;
				k=new CCommWordKernel(size);

				if (k)
				{
					CIO::message(M_INFO, "CommWordKernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"WORDSTRING")==0)
			{
				delete k;
				k=new CCommWordStringKernel(size);

				if (k)
				{
					CIO::message(M_INFO, "CommWordStringKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"FIXEDDEGREE")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT d=3;

				sscanf(param, "%s %s %d %d", kern_type, data_type, &size, &d);
				delete k;
				k=new CFixedDegreeCharKernel(size, d);

				if (k)
				{
					CIO::message(M_INFO, "FixedDegreeCharKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"WEIGHTEDDEGREEPOS")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT d=3;
				INT max_mismatch = 0 ;
				INT i=0;
				INT length = 0 ;
				INT center = 0 ;
				REAL step = 0 ;

				sscanf(param, "%s %s %d %d %d %d %d %le", 
					   kern_type, data_type, &size, &d, &max_mismatch, 
					   &length, &center, &step);
				CIO::message(M_INFO, "step = %le\n") ;
				
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
				
				INT *shift = new INT[length] ;
				for (INT i=center; i<length; i++)
					shift[i] = (int)floor(((REAL)(i-center))/step) ;

				for (INT i=center; i>=0; i--)
					shift[i] = (int)floor(((REAL)(center-i))/step) ;

				for (INT i=0; i<length; i++)
					if (shift[i]>length)
						shift[i]=length ;

				for (INT i=0; i<length; i++)
				  CIO::message(M_INFO, "shift[%i]=%i\n", i, shift[i]) ;
				
				delete k;
				k=new CWeightedDegreePositionCharKernel(size, weights, 
															 d, max_mismatch, 
															 shift, length);
				delete[] shift ;
				delete[] weights ;
				
				if (k)
				{
					CIO::message(M_INFO, "WeightedDegreePositionCharKernel(%d,.,%d,%d,.,%d) created\n",size, d, max_mismatch, length);
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"WEIGHTEDDEGREE")==0)
		{
			if (strcmp(data_type,"CHAR")==0)
			{
				INT d=3;
				INT max_mismatch = 0;
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
							INT nk=math.nchoosek(i+1, j);
							weights[i+j*d]=weights[i]/(nk*pow(3,j));
						}
						else
							weights[i+j*d]= 0;
						
					}
				}
				
				delete k;
				k=new CWeightedDegreeCharKernel(size, weights, d, max_mismatch);
				delete[] weights ;
				
				if (k)
				{
					CIO::message(M_INFO, "WeightedDegreeCharKernel created\n");
					return k;
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
				delete k;
				k=new CSimpleLocalityImprovedCharKernel(size, l, d1, d2);
				if (k)
				{
					CIO::message(M_INFO, "SimpleLocalityImprovedCharKernel created\n");
					return k;
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
				delete k;
				k=new CLocalityImprovedCharKernel(size, l, d1, d2);
				if (k)
				{
					CIO::message(M_INFO, "LocalityImprovedCharKernel created\n");
					return k;
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
				delete k;
				k=new CPolyKernel(size, degree, inhomogene==1);

				if (k)
				{
					CIO::message(M_INFO, "Polynomial Kernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"CHAR")==0)
			{
				INT inhomogene=0;
				INT degree=2;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &degree, &inhomogene);
				delete k;
				k=new CCharPolyKernel(size, degree, inhomogene==1);

				if (k)
				{
					CIO::message(M_INFO, "CharPolynomial Kernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				INT inhomogene=0;
				INT degree=2;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &degree, &inhomogene);
				delete k;
				k=new CSparsePolyKernel(size, degree, inhomogene==1);

				if (k)
				{
					CIO::message(M_INFO, "Sparse Polynomial Kernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"GAUSSIAN")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				double width=1;

				sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &width);
				delete k;
				k=new CGaussianKernel(size, width);
				if (k)
				{
					CIO::message(M_INFO, "Gaussian Kernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				double width=1;

				sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &width);
				delete k;
				k=new CSparseGaussianKernel(size, width);
				if (k)
				{
					CIO::message(M_INFO, "Sparse Gaussian Kernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"NORMSQUARED")==0)
		{
			if (strcmp(data_type,"SPARSEREAL")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CSparseNormSquaredKernel(size);
				if (k)
				{
					CIO::message(M_INFO, "Sparse NormSquared Kernel created\n");
					return k;
				}
			}
		}
		else 
			CIO::not_implemented();
	}
	else 
		CIO::message(M_ERROR, "see help for params!\n");

	CIO::not_implemented();
	return NULL;
}

bool CGUIKernel::add_kernel(CHAR* param)
{
	if ((kernel==NULL) || (kernel && kernel->get_kernel_type()!=K_COMBINED))
	{
		delete kernel;
		kernel= new CCombinedKernel(20);
		assert(kernel);
	}

	if (kernel)
	{
		char newparam[1000] ;
		double weight=1 ;
		
		int ret = sscanf(param, "%le %[a-zA-Z _*/+-0-9]", &weight, newparam) ;
		
		if (ret!=2)
		{
			CIO::message(M_ERROR, "add_kernel <weight> <kernel-parameters>\n");
			return false ;
		}

		CKernel* k=create_kernel(newparam);
		k->set_combined_kernel_weight(weight) ;
		
		assert(k);
		bool bret = ((CCombinedKernel*) kernel)->append_kernel(k) ;
		assert(bret);
		((CCombinedKernel*) kernel)->list_kernels();
		return true;
	}
	else
		CIO::message(M_ERROR, "combined kernel object could not be created\n");

	return false;
}

bool CGUIKernel::del_kernel(CHAR* param)
{
	return false;
}
