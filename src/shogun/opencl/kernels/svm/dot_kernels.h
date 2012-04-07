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
		typedef std::map<EKernelType, const char *> SourcesMapType;
		static SourcesMapType create_sources_map();
	public:
		static std::string dot_kernel_name(EFeatureType feature_type);
		static std::string program_name();
		static void init();
	};
	
    }
    
  }
  
}

#endif