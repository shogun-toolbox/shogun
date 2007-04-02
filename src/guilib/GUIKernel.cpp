/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "gui/GUI.h"
#include "guilib/GUIKernel.h"
#include "guilib/GUISVM.h"
#include "guilib/GUIPluginEstimate.h"
#include "kernel/Kernel.h"
#include "kernel/CombinedKernel.h"
#include "kernel/Chi2Kernel.h"
#include "kernel/LinearKernel.h"
#include "kernel/LinearByteKernel.h"
#include "kernel/LinearCharKernel.h"
#include "kernel/LinearWordKernel.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "kernel/WeightedDegreeStringKernel.h"
#include "kernel/WeightedDegreeCharKernelPolyA.h"
#include "kernel/WeightedDegreePositionCharKernel.h"
#include "kernel/WeightedDegreePositionStringKernel.h"
#include "kernel/WeightedDegreePositionPhylCharKernel.h"
#include "kernel/FixedDegreeCharKernel.h"
#include "kernel/LocalityImprovedStringKernel.h"
#include "kernel/SimpleLocalityImprovedCharKernel.h"
#include "kernel/PolyKernel.h"
#include "kernel/CustomKernel.h"
#include "kernel/ConstKernel.h"
#include "kernel/PolyMatchWordKernel.h"
#include "kernel/PolyMatchCharKernel.h"
#include "kernel/WordMatchKernel.h"
#include "kernel/CommWordKernel.h"
#include "kernel/HammingWordKernel.h"
#include "kernel/ManhattenWordKernel.h"
#include "kernel/CanberraWordKernel.h"
#include "kernel/CommWordStringKernel.h"
#include "kernel/CommUlongStringKernel.h"
#include "kernel/HistogramWordKernel.h"
#include "kernel/SalzbergWordKernel.h"
#include "kernel/GaussianKernel.h"
#include "kernel/SigmoidKernel.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SparseGaussianKernel.h"
#include "kernel/SparseNormSquaredKernel.h"
#include "kernel/DiagKernel.h"
#include "kernel/MindyGramKernel.h"
#include "kernel/DistanceKernel.h"
#include "lib/io.h"
#include "gui/GUI.h"

#include <string.h>

CGUIKernel::CGUIKernel(CGUI * gui_): CSGObject(), gui(gui_)
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
		SG_ERROR( "kernel creation failed.\n");
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
				SG_ERROR( "reading from file %s failed!\n", filename);
			else
			{
				SG_INFO( "successfully read kernel init data from \"%s\" !\n", filename);
				initialized=true;
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			SG_ERROR( "see help for params\n");
	}
	else
		SG_ERROR( "no kernel set!\n");
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
				SG_ERROR( "fname: %s\n", filename);
			if ((!file) || (!kernel->save_init(file)))
				SG_ERROR( "writing to file %s failed!\n", filename);
			else
			{
				SG_INFO( "successfully written kernel init data into \"%s\" !\n", filename);
				result=true;
			}

			if (file)
				fclose(file);
		}
		else
			SG_ERROR( "see help for params\n");
	}
	else
		SG_ERROR( "no kernel set!\n");
	return result;
}

bool CGUIKernel::init_kernel_optimization(CHAR* param)
{

	kernel->set_precompute_matrix(false, false);

	if (gui->guisvm.get_svm()!=NULL)
	{
		if (kernel->has_property(KP_LINADD))
		{
			INT * sv_idx    = new INT[gui->guisvm.get_svm()->get_num_support_vectors()] ;
			DREAL* sv_weight = new DREAL[gui->guisvm.get_svm()->get_num_support_vectors()] ;
			
			for(INT i=0; i<gui->guisvm.get_svm()->get_num_support_vectors(); i++)
			{
				sv_idx[i]    = gui->guisvm.get_svm()->get_support_vector(i) ;
				sv_weight[i] = gui->guisvm.get_svm()->get_alpha(i) ;
			}

			bool ret = kernel->init_optimization(gui->guisvm.get_svm()->get_num_support_vectors(), sv_idx, sv_weight) ;
			
			delete[] sv_idx ;
			delete[] sv_weight ;

			if (!ret)
				SG_ERROR( "initialization of kernel optimization failed\n") ;
			return ret ;
		}

	}
	else
	{
		SG_ERROR( "create SVM first\n");
		return false ;
	}
	return true ;
}

bool CGUIKernel::delete_kernel_optimization(CHAR* param)
{
	if (kernel && kernel->has_property(KP_LINADD) && kernel->get_is_initialized())
		kernel->delete_optimization() ;

	return true ;
}


bool CGUIKernel::init_kernel(CHAR* param)
{
	CHAR target[1024]="";

	if (!kernel)
	{
		SG_ERROR( "no kernel available\n") ;
		return false ;
	} ;

	kernel->set_precompute_matrix(false, false);

	if ((sscanf(param, "%s", target))==1)
	{
		if (!strncmp(target, "TRAIN", 5))
		{
			if (gui->guifeatures.get_train_features())
			{
				if ( (kernel->get_feature_class() == gui->guifeatures.get_train_features()->get_feature_class() 
							|| gui->guifeatures.get_train_features()->get_feature_class() == C_ANY 
							|| kernel->get_feature_class() == C_ANY ) &&
						(kernel->get_feature_type() == gui->guifeatures.get_train_features()->get_feature_type() 
						 || gui->guifeatures.get_train_features()->get_feature_type() == F_ANY 
						 || kernel->get_feature_type() == F_ANY) )
				{
					kernel->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_train_features());
					initialized=true;
				}
				else
				{
					SG_ERROR( "kernel can not process this feature type\n");
					return false ;
				}
			}
			else
				SG_ERROR( "assign train features first\n");
		}
		else if (!strncmp(target, "TEST", 5))
		{
			if (gui->guifeatures.get_train_features() && gui->guifeatures.get_test_features())
			{
				if ( (kernel->get_feature_class() == gui->guifeatures.get_train_features()->get_feature_class() 
							|| gui->guifeatures.get_train_features()->get_feature_class() == C_ANY 
							|| kernel->get_feature_class() == C_ANY ) &&
						(kernel->get_feature_class() == gui->guifeatures.get_test_features()->get_feature_class() 
							|| gui->guifeatures.get_test_features()->get_feature_class() == C_ANY 
							|| kernel->get_feature_class() == C_ANY ) &&
						(kernel->get_feature_type() == gui->guifeatures.get_train_features()->get_feature_type() 
						 || gui->guifeatures.get_train_features()->get_feature_type() == F_ANY 
						 || kernel->get_feature_type() == F_ANY ) &&
						(kernel->get_feature_type() == gui->guifeatures.get_test_features()->get_feature_type() 
						 || gui->guifeatures.get_test_features()->get_feature_type() == F_ANY 
						 || kernel->get_feature_type() == F_ANY ) )
				{
					if (!initialized)
					{
						SG_ERROR( "kernel not initialized for training examples\n") ;
						return false ;
					}
					else
					{
						SG_INFO( "initialising kernel with TEST DATA, train: %d test %d\n",gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features() );
						// lhs -> always train_features; rhs -> always test_features
						kernel->init(gui->guifeatures.get_train_features(), gui->guifeatures.get_test_features());
					} ;
				}
				else
				{
					SG_ERROR( "kernel can not process this feature type\n");
					return false ;
				}
			}
			else
				SG_ERROR( "assign train and test features first\n");

		}
		else
			io.not_implemented();
	}
	else 
	{
		SG_ERROR( "see help for params\n");
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
				SG_ERROR( "writing to file %s failed!\n", filename);
			else
			{
				SG_INFO( "successfully written kernel to \"%s\" !\n", filename);
				result=true;
			}
		}
		else
			SG_ERROR( "see help for params\n");
	}
	else
		SG_ERROR( "no kernel set / kernel not initialized!\n");
	return result;
}

CKernel* CGUIKernel::create_kernel(CHAR* param)
{
	INT size=100;
	CHAR kern_type[1024]="";
	CHAR data_type[1024]="";
	param=CIO::skip_spaces(param);
	CKernel* k=NULL;
	INT append_subkernel_weights = 0 ;

	//note the different args COMBINED <cachesize>
	if (sscanf(param, "%s %d %i", kern_type, &size, &append_subkernel_weights) == 3)
	{
		if (strcmp(kern_type,"COMBINED")==0)
		{
			delete k;
			k= new CCombinedKernel(size, append_subkernel_weights!=0);
			if (kernel)
				SG_INFO( "CombinedKernel created\n");
			return k;
		}
		else if(strcmp(kern_type,"DISTANCE")==0)
                {
			double width = 1.0;
                        delete k;
			
			CDistance* dist = gui->guidistance.get_distance();
			
			if(dist)
			{
				sscanf(param,"%s %d %lf",kern_type,&size,&width);
				k = new CDistanceKernel(size,width,dist);
				SG_INFO("DistanceKernel created");				
				return k;
			}
			else
			{
				SG_ERROR("no distance set for DistanceKernel");
			}

		} 
		else
			SG_ERROR( "in this format I only expect Combined kernels\n") ;
	} 
	else if (sscanf(param, "%s %d", kern_type, &size) == 2)
	{
		if (strcmp(kern_type,"COMBINED")==0)
		{
			delete k;
			k= new CCombinedKernel(size,false);
			if (kernel)
				SG_INFO( "CombinedKernel created\n");
			return k;
		} else
			SG_ERROR( "in this format I only expect Combined kernels\n") ;
	} 
	//compared with <KERNTYPE> <DATATYPE> <CACHESIZE>
	else if (sscanf(param, "%s %s %d", kern_type, data_type, &size) >= 2)
	{
		if (strcmp(kern_type,"LINEAR")==0)
		{
			if (strcmp(data_type,"BYTE")==0)
			{
				double scale = -1 ;
				sscanf(param, "%s %s %d %le", kern_type, data_type, &size, &scale);
				delete k;
				if (scale==-1)
					k=new CLinearWordKernel(size, true);
				else
					k=new CLinearWordKernel(size, false, scale);
				return k;
			}
			else if (strcmp(data_type,"WORD")==0)
			{
				double scale = -1 ;
				sscanf(param, "%s %s %d %le", kern_type, data_type, &size, &scale);
				delete k;
				if (scale==-1)
					k=new CLinearWordKernel(size, true);
				else
					k=new CLinearWordKernel(size, false, scale);
				return k;
			}
			else if (strcmp(data_type,"CHAR")==0)
			{
				double scale = -1 ;
				sscanf(param, "%s %s %d %le", kern_type, data_type, &size, &scale);
				delete k;
				if (scale==-1)
					k=new CLinearCharKernel(size, true);
				else
					k=new CLinearCharKernel(size, false, scale);
				return k;
			}
			else if (strcmp(data_type,"REAL")==0)
			{
				double scale = 0;
				sscanf(param, "%s %s %d %le", kern_type, data_type, &size, &scale);
				delete k;
				k=new CLinearKernel(size, scale);
				return k;
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				double scale = -1 ;
				sscanf(param, "%s %s %d %le", kern_type, data_type, &size, &scale);
				delete k;
				if (scale==-1)
					k=new CSparseLinearKernel(size, true);
				else
					k=new CSparseLinearKernel(size, false, scale);
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
				    SG_INFO( "destroying old k\n") ;
				    delete k;
				  } ;

				SG_INFO( "getting estimator\n") ;
				CPluginEstimate* estimator=gui->guipluginestimate.get_estimator();

				if (estimator)
					k=new CHistogramWordKernel(size, estimator);
				else
					SG_ERROR( "no estimator set\n");

				if (k)
				{
					SG_INFO( "HistogramKernel created\n");
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
				    SG_INFO( "destroying old k\n") ;
				    delete k;
				  } ;

				SG_INFO( "getting estimator\n") ;
				CPluginEstimate* estimator=gui->guipluginestimate.get_estimator();

				SG_INFO( "getting labels\n") ;
				CLabels * train_labels = gui->guilabels.get_train_labels() ;
				if (!train_labels)
				{
					SG_INFO( "assign train labels first!\n") ;
					return NULL ;
				} ;
				
				INT num_pos=0, num_neg=0;
				
				for (INT i=0; i<train_labels->get_num_labels(); i++)
				{
					if (train_labels->get_int_label(i)==1) num_pos++ ;
					if (train_labels->get_int_label(i)==-1) num_neg++ ;
				}				
				SG_INFO( "priors: pos=%1.3f (%i)  neg=%1.3f (%i)\n", 
							 (DREAL) num_pos/(num_pos+num_neg), num_pos,
							 (DREAL) num_neg/(num_pos+num_neg), num_neg) ;
				
				if (estimator)
					k=new CSalzbergWordKernel(size, estimator);
				else
					SG_ERROR( "no estimator set\n");
				
				((CSalzbergWordKernel*)k)->set_prior_probs((DREAL)num_pos/(num_pos+num_neg), 
																(DREAL)num_neg/(num_pos+num_neg)) ;
				
				if (k)
				{
					SG_INFO( "SalzbergKernel created\n");
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
				INT normalize=1;

				sscanf(param, "%s %s %d %d %d %d", kern_type, data_type, &size, &degree, &inhomogene, &normalize);
				delete k;
				k=new CPolyMatchWordKernel(size, degree, inhomogene==1, normalize==1);

				if (k)
				{
					SG_INFO( "PolyMatchWordKernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"CHAR")==0)
			{
				INT inhomogene=0;
				INT degree=2;
				INT normalize=1;

				sscanf(param, "%s %s %d %d %d %d", kern_type, data_type, &size, &degree, &inhomogene, &normalize);
				delete k;
				k=new CPolyMatchCharKernel(size, degree, inhomogene==1, normalize==1);

				if (k)
				{
					SG_INFO( "PolyMatchCharKernel created\n");
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
					SG_INFO( "MatchKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"COMMSTRING")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				delete k;
				INT use_sign = 0 ;
				char normalization_str[100]="" ;
				ENormalizationType normalization = FULL_NORMALIZATION ;
				
				sscanf(param, "%s %s %d %d %s", kern_type, data_type, &size, &use_sign, normalization_str);
				if (strlen(normalization_str)==0)
				{
					normalization = FULL_NORMALIZATION ;
					SG_INFO( "using full normalization\n") ;
				}
				else if (strcmp(normalization_str, "NO")==0)
				{
					normalization = NO_NORMALIZATION ;
					SG_INFO( "using no normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQRT")==0)
				{
					normalization = SQRT_NORMALIZATION ;
					SG_INFO( "using sqrt normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQRTLEN")==0)
				{
					normalization = SQRTLEN_NORMALIZATION ;
					SG_INFO( "using sqrt-len normalization\n") ;
				}
				else if (strcmp(normalization_str, "LEN")==0)
				{
					normalization = LEN_NORMALIZATION ;
					SG_INFO( "using len normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQLEN")==0)
				{
					normalization = SQLEN_NORMALIZATION ;
					SG_INFO( "using squared len normalization\n") ;
				}
				else if (strcmp(normalization_str, "FULL")==0)
				{
					normalization = FULL_NORMALIZATION ;
					SG_INFO( "using full normalization\n") ;
				}
				else 
				{
					SG_ERROR( "unknow normalization: %s\n", normalization_str) ;
					return NULL ;
				}

				k=new CCommWordStringKernel(size, use_sign==1, normalization);
				
				if (k)
				{
					if (use_sign)
						SG_INFO( "CommWordStringKernel with sign(count) created\n");
					else
						SG_INFO( "CommWordStringKernel with count created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"ULONG")==0)
			{
				delete k;
				INT use_sign = 0 ;
				char normalization_str[100]="" ;
				ENormalizationType normalization = FULL_NORMALIZATION ;
				
				sscanf(param, "%s %s %d %d %s", kern_type, data_type, &size, &use_sign, normalization_str);
				if (strlen(normalization_str)==0)
				{
					normalization = FULL_NORMALIZATION ;
					SG_INFO( "using full normalization\n") ;
				}
				else if (strcmp(normalization_str, "NO")==0)
				{
					normalization = NO_NORMALIZATION ;
					SG_INFO( "using no normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQRT")==0)
				{
					normalization = SQRT_NORMALIZATION ;
					SG_INFO( "using sqrt normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQRTLEN")==0)
				{
					normalization = SQRTLEN_NORMALIZATION ;
					SG_INFO( "using sqrt-len normalization\n") ;
				}
				else if (strcmp(normalization_str, "LEN")==0)
				{
					normalization = LEN_NORMALIZATION ;
					SG_INFO( "using len normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQLEN")==0)
				{
					normalization = SQLEN_NORMALIZATION ;
					SG_INFO( "using squared len normalization\n") ;
				}
				else if (strcmp(normalization_str, "FULL")==0)
				{
					normalization = FULL_NORMALIZATION ;
					SG_INFO( "using full normalization\n") ;
				}
				else 
				{
					SG_ERROR( "unknow normalization: %s\n", normalization_str) ;
					return NULL ;
				}

				k=new CCommUlongStringKernel(size, use_sign, normalization);
				
				if (k)
				{
					if (use_sign)
						SG_INFO( "CommUlongStringKernel with sign(count) created\n");
					else
						SG_INFO( "CommUlongStringKernel with count created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"COMM")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				INT use_sign = 0 ;
				char normalization_str[100]="" ;
				ENormalizationType normalization = FULL_NORMALIZATION ;
				
				sscanf(param, "%s %s %d %d %s", kern_type, data_type, &size, &use_sign, normalization_str);
				if (strlen(normalization_str)==0)
				{
					normalization = FULL_NORMALIZATION ;
					SG_INFO( "using full normalization\n") ;
				}
				else if (strcmp(normalization_str, "NO")==0)
				{
					normalization = NO_NORMALIZATION ;
					SG_INFO( "using no normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQRT")==0)
				{
					normalization = SQRT_NORMALIZATION ;
					SG_INFO( "using sqrt normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQRTLEN")==0)
				{
					normalization = SQRTLEN_NORMALIZATION ;
					SG_INFO( "using sqrt-len normalization\n") ;
				}
				else if (strcmp(normalization_str, "LEN")==0)
				{
					normalization = LEN_NORMALIZATION ;
					SG_INFO( "using len normalization\n") ;
				}
				else if (strcmp(normalization_str, "SQLEN")==0)
				{
					normalization = SQLEN_NORMALIZATION ;
					SG_INFO( "using squared len normalization\n") ;
				}
				else if (strcmp(normalization_str, "FULL")==0)
				{
					normalization = FULL_NORMALIZATION ;
					SG_INFO( "using full normalization\n") ;
				}
				else 
				{
					SG_ERROR( "unknow normalization: %s\n", normalization_str) ;
					return NULL ;
				}

				delete k;
				k=new CCommWordKernel(size, use_sign==1, normalization);

				if (k)
				{
					if (use_sign)
						SG_INFO( "CommWordKernel with sign(count) created\n");
					else
						SG_INFO( "CommWordKernel with count created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"HAMMING")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				INT use_sign = 0 ;
				double width=1.0;
				
				sscanf(param, "%s %s %d %lf %d", kern_type, data_type, &size, &width, &use_sign);
				delete k;
				k=new CHammingWordKernel(size, width, use_sign==1);

				if (k)
				{
					if (use_sign)
						SG_INFO( "HammingWordKernel with sign(count) created\n");
					else
						SG_INFO( "HammingWordKernel with count created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"MANHATTEN")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				double width=1.0;
				
				sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &width);
				delete k;
				k=new CManhattenWordKernel(size, width);

				if (k)
				{
					SG_INFO( "ManhattenWordKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"CANBERRA")==0)
		{
			if (strcmp(data_type,"WORD")==0)
			{
				double width=1.0;
				
				sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &width);
				delete k;
				k=new CCanberraWordKernel(size, width);

				if (k)
				{
					SG_INFO( "CanberraWordKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"CHI2")==0)
		{
			if (strcmp(data_type,"REAL")==0)
			{
				sscanf(param, "%s %s %d", kern_type, data_type, &size);
				delete k;
				k=new CChi2Kernel(size);

				if (k)
				{
					SG_INFO( "Chi2Kernel created\n");
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
					SG_INFO( "FixedDegreeCharKernel created\n");
					return k;
				}
			}
		}
		else if ((strcmp(kern_type,"WEIGHTEDDEGREEPOS2")==0) || (strcmp(kern_type,"WEIGHTEDDEGREEPOSPHYL2")==0))
		{
			if ((strcmp(data_type,"CHAR")==0) || (strcmp(data_type,"STRING")==0))
			{
				INT d=3;
				INT max_mismatch = 0 ;
				INT i=0;
				INT length = 0 ;
				char * rest = new char[strlen(param)] ;
				char * rest2 = new char[strlen(param)] ;
				
				sscanf(param, "%s %s %d %d %d %d %[0-9 .+-]", 
					   kern_type, data_type, &size, &d, &max_mismatch, 
					   &length, rest);

				INT *shift = new INT[length] ;
				for (i=0; i<length; i++)
				{
					int args = sscanf(rest, "%i %[0-9 .+-]", &shift[i], rest2) ;
					if (((args!=2) && (i<length-1)) || (args<1))
					{
						SG_ERROR( "failed to read list at position %i\n", i) ;
						return false ;
					} ;
					if (shift[i]>length)
					{
						SG_ERROR( "shift longer than sequence: %i \n", shift[i]) ;
						return false ;
					} ;
					strcpy(rest,rest2) ;
				}
				
				DREAL* weights=new DREAL[d*(1+max_mismatch)];
				DREAL sum=0;

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
							INT nk=CMath::nchoosek(i+1, j) ;
							weights[i+j*d]=weights[i]/(nk*pow(3,j)) ;
						}
						else
							weights[i+j*d]= 0;
					} ;
				} ;
				
				
				delete k;

				if (strcmp(kern_type,"WEIGHTEDDEGREEPOS2")==0)
				{
					if (strcmp(data_type,"CHAR")==0)
					{
						k=new CWeightedDegreePositionCharKernel(size, weights, 
								d, max_mismatch, 
								shift, length, true);
					}
					else
					{
						k=new CWeightedDegreePositionStringKernel(size, weights, 
								d, max_mismatch, 
								shift, length, true);
					}
				}
				else
				{
					if (strcmp(data_type,"CHAR")==0)
					{
						k=new CWeightedDegreePositionPhylCharKernel(size, weights, 
								d, max_mismatch, 
								shift, length, true);
					}
					else
						SG_ERROR("BROKEN\n");
				}

				delete[] shift ;
				delete[] weights ;
				
				if (k)
				{
					SG_INFO( "WeightedDegreePositionCharKernel(%d,.,%d,%d,.,%d) created\n",size, d, max_mismatch, length);
					return k;
				}
			}
		}
		else if ((strcmp(kern_type,"WEIGHTEDDEGREEPOS3")==0) || (strcmp(kern_type,"WEIGHTEDDEGREEPOSPHYL3")==0))
		{
			if ((strcmp(data_type,"CHAR")==0) || (strcmp(data_type,"STRING")==0))
			{
				INT d=3;
				INT max_mismatch = 0 ;
				INT i=0;
				INT length = 0 ;
				INT mkl_stepsize = 1 ;
				
				char * rest = new char[strlen(param)] ;
				char * rest2 = new char[strlen(param)] ;
				
				sscanf(param, "%s %s %d %d %d %d %d %[0-9 .+-]", 
					   kern_type, data_type, &size, &d, &max_mismatch, 
					   &length, &mkl_stepsize, rest);
				
				INT *shift = new INT[length] ;
				for (i=0; i<length; i++)
				{
					*rest2 = '\0';
					int args = sscanf(rest, "%i %[0-9 .+-]", &shift[i], rest2) ;
					if (((args!=2) && (i<length-1)) || (args<1))
					{
						SG_ERROR( "failed to read shift list at position %i\n", i) ;
						return false ;
					} ;
					if (shift[i]>length)
					{
						SG_ERROR( "shift longer than sequence: %i \n", shift[i]) ;
						return false ;
					} ;
					strcpy(rest,rest2) ;
				}
				
				DREAL *position_weights = new DREAL[length] ;
				for (INT ii=0; ii<length; ii++)
					position_weights[ii]=1.0/length ;
				if (strlen(rest)>0)
				{
					for (i=0; i<length; i++)
					{
						fprintf(stderr, "%i %s\n", i, rest) ;
						
						int args = sscanf(rest, "%le %[0-9 .+-]", &position_weights[i], rest2) ;
						if (((args!=2) && (i<length-1)) || (args<1))
						{
							if (i>0)
							{
								SG_ERROR( "failed to read weight list at position %i\n", i) ;
								return false ;
							}
							else break ;
						} ;
						if (position_weights[i]<0)
						{
							SG_ERROR( "no negative weights allowed: %1.1le \n", position_weights[i]) ;
							return false ;
						} ;
						strcpy(rest,rest2) ;
					}
				}
				
				DREAL* weights=new DREAL[d*(1+max_mismatch)];
				DREAL sum=0;

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
							INT nk=CMath::nchoosek(i+1, j) ;
							weights[i+j*d]=weights[i]/(nk*pow(3,j)) ;
						}
						else
							weights[i+j*d]= 0;
					} ;
				} ;
				
				
				delete k;
				
				if (strcmp(kern_type,"WEIGHTEDDEGREEPOS3")==0)
				{
					if (strcmp(data_type,"CHAR")==0)
					{
						k=new CWeightedDegreePositionCharKernel(size, weights, 
								d, max_mismatch, 
								shift, length, false, 
								mkl_stepsize);
					}
					else
					{
						k=new CWeightedDegreePositionStringKernel(size, weights, 
								d, max_mismatch, 
								shift, length, false, 
								mkl_stepsize);
					}
				}
				else
				{
					if (strcmp(data_type,"CHAR")==0)
					{
						k=new CWeightedDegreePositionPhylCharKernel(size, weights, 
								d, max_mismatch, 
								shift, length, false, 
								mkl_stepsize);
					}
					else
						SG_ERROR("BROKEN\n");
				}
				
				((CWeightedDegreePositionCharKernel*)k)->set_position_weights(position_weights, length) ;
				
				delete[] shift ;
				delete[] weights ;
				delete[] position_weights ;
				
				if (k)
				{
					SG_INFO( "WeightedDegreePositionCharKernel(%d,.,%d,%d,.,%d,%d) created\n",size, d, max_mismatch, length,mkl_stepsize);
					return k;
				}
			}
		}
		else if ((strcmp(kern_type,"WEIGHTEDDEGREEPOS")==0) || (strcmp(kern_type,"WEIGHTEDDEGREEPOSPHYL")==0))
		{
			if ((strcmp(data_type,"CHAR")==0) || (strcmp(data_type,"STRING")==0))
			{
				INT d=3;
				INT max_mismatch = 0 ;
				INT i=0;
				INT length = 0 ;
				INT center = 0 ;
				DREAL step = 0 ;

				sscanf(param, "%s %s %d %d %d %d %d %le", 
					   kern_type, data_type, &size, &d, &max_mismatch, 
					   &length, &center, &step);
				SG_INFO( "step = %le\n") ;
				
				DREAL* weights=new DREAL[d*(1+max_mismatch)];
				DREAL sum=0;

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
							INT nk=CMath::nchoosek(i+1, j) ;
							weights[i+j*d]=weights[i]/(nk*pow(3,j)) ;
						}
						else
							weights[i+j*d]= 0;
					} ;
				} ;
				
				INT *shift = new INT[length] ;
				for (INT ii=center; ii<length; ii++)
					shift[ii] = (int)floor(((DREAL)(ii-center))/step) ;

				for (INT ii=center; ii>=0; ii--)
					shift[ii] = (int)floor(((DREAL)(center-ii))/step) ;

				for (INT ii=0; ii<length; ii++)
					if (shift[ii]>length)
						shift[ii]=length ;

				for (INT ii=0; ii<length; ii++)
				  SG_INFO( "shift[%i]=%i\n", ii, shift[ii]) ;
				
				delete k;

				if (strcmp(kern_type,"WEIGHTEDDEGREEPOS")==0)
				{
					if (strcmp(data_type,"CHAR")==0)
					{
						k=new CWeightedDegreePositionCharKernel(size, weights, 
								d, max_mismatch, 
								shift, length);
					}
					else
					{
						k=new CWeightedDegreePositionStringKernel(size, weights, 
								d, max_mismatch, 
								shift, length);
					}
				}
				else
				{
					if (strcmp(data_type,"CHAR")==0)
					{
						k=new CWeightedDegreePositionCharKernel(size, weights, 
								d, max_mismatch, 
								shift, length);
					}
					else
						SG_ERROR("BROKEN\n");
				}
				delete[] shift ;
				delete[] weights ;
				
				if (k)
				{
					SG_INFO( "WeightedDegreePositionCharKernel(%d,.,%d,%d,.,%d) created\n",size, d, max_mismatch, length);
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"WEIGHTEDDEGREE")==0)
		{
			if ((strcmp(data_type,"CHAR")==0) || (strcmp(data_type,"STRING")==0))
			{
				INT use_normalization=1;
				INT d=3;
				INT max_mismatch = 0;
				INT i=0;
				INT mkl_stepsize = 1 ;
				INT block_computation = 1;
				INT single_degree = -1;

				sscanf(param, "%s %s %d %d %d %d %d %d %d", kern_type, data_type, &size, &d, &max_mismatch, &use_normalization, &mkl_stepsize, &block_computation, &single_degree);
				DREAL* weights=new DREAL[d*(1+max_mismatch)];
				DREAL sum=0;

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
							INT nk=CMath::nchoosek(i+1, j);
							weights[i+j*d]=weights[i]/(nk*pow(3,j));
						}
						else
							weights[i+j*d]= 0;
						
					}
				}

				if (single_degree>=0)
				{
					ASSERT(single_degree<d);
					for (i=0; i<d; i++)
					{
						if (i!=single_degree)
							weights[i]=0;
						else
							weights[i]=1;
					}
				}

				delete k;
				if (strcmp(data_type,"CHAR")==0)
				{
					k=new CWeightedDegreeCharKernel(size, weights, d, max_mismatch, use_normalization==1, block_computation==1, mkl_stepsize);
				}
				else
				{
					k=new CWeightedDegreeStringKernel(size, weights, d, max_mismatch, use_normalization==1, block_computation==1, mkl_stepsize);
				}

				delete[] weights ;
				
				if (k)
				{
					SG_INFO( "WeightedDegreeCharKernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"WEIGHTEDDEGREEPOLYA")==0)
		{
			if ((strcmp(data_type,"CHAR")==0) || (strcmp(data_type,"STRING")==0))
			{
				INT d=3;
				INT max_mismatch = 0;
				INT i=0;

				sscanf(param, "%s %s %d %d %d", kern_type, data_type, &size, &d, &max_mismatch);
				DREAL* weights=new DREAL[d*(1+max_mismatch)];
				DREAL sum=0;

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
							INT nk=CMath::nchoosek(i+1, j);
							weights[i+j*d]=weights[i]/(nk*pow(3,j));
						}
						else
							weights[i+j*d]= 0;
						
					}
				}
				
				delete k;
				if (strcmp(data_type,"CHAR")==0)
					k=new CWeightedDegreeCharKernelPolyA(size, weights, d, max_mismatch);
				else
					SG_ERROR("BROKEN");
				delete[] weights ;
				
				if (k)
				{
					SG_INFO( "WeightedDegreeCharKernelPolyA created\n");
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
					SG_INFO( "SimpleLocalityImprovedCharKernel created\n");
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
				k=new CLocalityImprovedStringKernel(size, l, d1, d2);
				if (k)
				{
					SG_INFO( "LocalityImprovedStringKernel created\n");
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
				INT normalize=1;

				sscanf(param, "%s %s %d %d %d %d", kern_type, data_type, &size, &degree, &inhomogene, &normalize);
				delete k;
				k=new CPolyKernel(size, degree, inhomogene==1, normalize==1);

				if (k)
				{
					SG_INFO( "Polynomial Kernel created\n");
					return k;
				}
			}
			else if (strcmp(data_type,"SPARSEREAL")==0)
			{
				INT inhomogene=0;
				INT degree=2;
				INT normalize=1;

				sscanf(param, "%s %s %d %d %d %d", kern_type, data_type, &size, &degree, &inhomogene, &normalize);
				delete k;
				k=new CSparsePolyKernel(size, degree, inhomogene==1, normalize==1);

				if (k)
				{
					SG_INFO( "Sparse Polynomial Kernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"SIGMOID")==0) // tanh
		{
			if (strcmp(data_type,"REAL")==0)
			{
				double gamma=0.01;
				double coef0=0;

				sscanf(param, "%s %s %d %lf %lf", kern_type, data_type, &size, &gamma, &coef0);
				delete k;
				k=new CSigmoidKernel(size, gamma, coef0);
				if (k)
				{
					SG_INFO( "Sigmoid Kernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"GAUSSIAN")==0) // RBF
		{
			if (strcmp(data_type,"REAL")==0)
			{
				double width=1;

				sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &width);
				delete k;
				k=new CGaussianKernel(size, width);
				if (k)
				{
					SG_INFO( "Gaussian Kernel created\n");
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
					SG_INFO( "Sparse Gaussian Kernel created\n");
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
					SG_INFO( "Sparse NormSquared Kernel created\n");
					return k;
				}
			}
		}
		else if (strcmp(kern_type,"CUSTOM")==0)
		{
			delete k;
			k = new CCustomKernel();
			if (k)
			{
				SG_INFO( "Custom Kernel created\n");
				return k;
			}
		}
		else if (strcmp(kern_type,"CONST")==0)
		{
			delete k;
			DREAL val=0;
			sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &val);
			k = new CConstKernel(val);
			if (k)
			{
				SG_INFO( "Const Kernel created\n");
				return k;
			}
		}
		else if (strcmp(kern_type,"DIAG")==0)
		{
			DREAL diag=1.0;

			sscanf(param, "%s %s %d %lf", kern_type, data_type, &size, &diag);
			delete k;
			k = new CDiagKernel(size, diag);
			if (k)
			{
				SG_INFO( "Diag Kernel created\n");
				return k;
			}
		}
#ifdef HAVE_MINDY
		else if (strcmp(kern_type,"MINDYGRAM")==0)
                {
                        delete k;
                        char norm_str[256]="";
                        char param_str[256]="";
                        char meas_str[256]="";
                        float width;

                        ENormalizationType normalization = FULL_NORMALIZATION ;

                        sscanf(param, "%s %s %d %255s %255s %f %255s", kern_type, data_type, &size, meas_str, norm_str, &width, param_str);
                        if (strlen(norm_str)==0)
                        {
                                normalization = FULL_NORMALIZATION ;
                                SG_INFO( "using full normalization (default)\n") ;
                        }
                        else if (strcmp(norm_str, "NO")==0)
                        {
                                normalization = NO_NORMALIZATION ;
                                SG_INFO( "using no normalization\n") ;
                        }
                        else if (strcmp(norm_str, "SQRT")==0)
                        {
                                normalization = SQRT_NORMALIZATION ;
                                SG_INFO( "using sqrt normalization\n") ;
                        }
                        else if (strcmp(norm_str, "FULL")==0)
                        {
                                normalization = FULL_NORMALIZATION ;
                                SG_INFO( "using full normalization\n") ;
                        }
                        else
                        {
                                SG_ERROR( "unknow normalization: %s\n", norm_str) ;
                                return NULL ;
                        }

                        CMindyGramKernel* mk = new CMindyGramKernel(size, meas_str, width);
                        if (!mk) {
                        	SG_ERROR("could not allocate kernel object");
                        }
                        mk->set_norm(normalization);
                        mk->set_param(param_str);
                        
                        return mk;
		}
#endif
		else
			io.not_implemented();
	}
	else 
		SG_ERROR( "see help for params!\n");

	io.not_implemented();
	return NULL;
}

bool CGUIKernel::add_kernel(CHAR* param)
{
	if ((kernel==NULL) || (kernel && kernel->get_kernel_type()!=K_COMBINED))
	{
		delete kernel;
		kernel= new CCombinedKernel(20,false);
		ASSERT(kernel);
	}

	if (kernel)
	{
		CHAR *newparam=new CHAR[strlen(param)] ;
		double weight=1 ;
		
		int ret = sscanf(param, "%le %[a-zA-Z _*/+-0-9]", &weight, newparam) ;
		//fprintf(stderr,"ret=%i  weight=%le  rest=%s\n", ret, weight, newparam) ;
		
		if (ret!=2)
		{
			SG_ERROR( "add_kernel <weight> <kernel-parameters>\n");
			delete[] newparam ;
			return false ;
		}

		CKernel* k=create_kernel(newparam);
		ASSERT(k);
		k->set_combined_kernel_weight(weight) ;
		
		bool bret = ((CCombinedKernel*) kernel)->append_kernel(k) ;
		if (bret)
			((CCombinedKernel*) kernel)->list_kernels();
		else
			SG_ERROR( "appending kernel failed\n");

		delete[] newparam ;
		return bret;
	}
	else
		SG_ERROR( "combined kernel object could not be created\n");

	return false;
}

bool CGUIKernel::clean_kernel(CHAR* param)
{
	delete kernel;
	kernel = NULL;
	return true;
}

#ifdef USE_SVMLIGHT
bool CGUIKernel::resize_kernel_cache(CHAR* param)
{
	if (kernel!=NULL) 
	{
		INT size = 10 ;
		sscanf(param, "%d", &size);
		kernel->resize_kernel_cache(size) ;
		return true ;
	}
	SG_ERROR( "no kernel available\n") ;
	return false;
}
#endif //USE_SVMLIGHT

bool CGUIKernel::set_optimization_type(CHAR* param)
{
	EOptimizationType opt=SLOWBUTMEMEFFICIENT;
	char opt_type[1024];
	param=CIO::skip_spaces(param);

	if (kernel!=NULL) 
	{
		if (sscanf(param, "%s", opt_type)==1)
		{
			if (strcmp(opt_type,"FASTBUTMEMHUNGRY")==0)
			{
				SG_INFO("FAST METHOD selected\n");
				opt=FASTBUTMEMHUNGRY;
				kernel->set_optimization_type(opt);
				return true;
			}
			else if (strcmp(opt_type,"SLOWBUTMEMEFFICIENT")==0)
			{
				SG_INFO("MEMORY EFFICIENT METHOD selected\n");
				opt=SLOWBUTMEMEFFICIENT;
				kernel->set_optimization_type(opt);
				return true;
			}
			else
				SG_ERROR( "option missing\n");
		}
	}
	SG_ERROR( "no kernel available\n") ;
	return false;
}

bool CGUIKernel::del_kernel(CHAR* param)
{
	return false;
}
#endif
