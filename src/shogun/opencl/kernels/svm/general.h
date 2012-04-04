#ifndef SVM_GENERAL_KERNELS_HPP
#define SVM_GENERAL_KERNELS_HPP

#include <shogun/opencl/kernels/stringify.h>
#include <viennacl/ocl/utils.hpp>
#include <shogun/features/FeatureTypes.h>
#include <shogun/kernel/Kernel.h>
#include <map>




namespace shogun{
  
  namespace ocl{
    
    namespace svm{
	    
	class general{
      
	private:
		typedef std::map<const char *, const char *> SourcesMapType;
		static SourcesMapType create_sources_map();
	public:
		static std::string program_name();
		static void init();
	};
	
    }
    
  }
  
}

#endif