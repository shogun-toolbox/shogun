#ifndef SVM_DOT_KERNELS_HPP
#define SVM_DOT_KERNELS_HPP

#include <shogun/opencl/kernels/stringify.h>
#include <viennacl/ocl/utils.hpp>
#include <shogun/features/FeatureTypes.h>
#include <shogun/kernel/Kernel.h>
#include <map>




namespace shogun{
  
  namespace ocl{
    
    namespace svm{
	    
	class dot_kernels{
	  
	private:
		typedef std::map<std::pair<EKernelType,EFeatureType>, const char *> SourcesMapType;
		static SourcesMapType create_sources_map();
		static std::string dot_kernel_src(EFeatureType feature_type);
	public:	
		static std::string program_name(EFeatureType feature_type);
		static void init(EFeatureType feature_type);
	};
	
    }
    
  }
  
}

#endif