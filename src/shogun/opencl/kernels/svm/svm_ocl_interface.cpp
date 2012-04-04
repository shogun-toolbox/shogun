#include "dot_kernels.h"
#include "general.h"
#include <shogun/opencl/viennacl_compatibility.h>

namespace shogun{
  
  namespace ocl{
    
	bool is_ocl_valid(EFeatureType feature_type){
		return feature_type == F_SHORTREAL
			||feature_type == F_LONGREAL;
	}
		
    namespace svm{
	    		
	dot_kernels::SourcesMapType dot_kernels::create_sources_map()
	{
		SourcesMapType m;
		m[std::make_pair(K_GAUSSIAN,F_SHORTREAL)] = 
		#include "dot_kernels/float/gaussian_kernel.cl"
				;
		return m;
	}
	
		
	std::string dot_kernels::program_name(EFeatureType feature_type){
		if(feature_type==F_SHORTREAL){
			return "svm_dot_float";
		}
		if(feature_type==F_LONGREAL){
			return "svm_dot_doule";
		}
		return "";
	}

	std::string dot_kernels::dot_kernel_src(EFeatureType feature_type){
		if(feature_type==F_SHORTREAL){
			std::string res;
			res =
			#include "dot_kernels/float/dot_kernel.cl"
			;
			return res;
		}
		if(feature_type==F_LONGREAL){
			//TODO
			return "";
		}
		return "";
	}
	
	void dot_kernels::init(EFeatureType feature_type){
		if(!is_ocl_valid(feature_type)) return;
		std::cout << "Init" << std::endl;
		typedef std::pair<EFeatureType, cl_context> KeyType;
		static std::map<KeyType, bool> init_done;	
		static SourcesMapType ocl_sources_map = create_sources_map();
		viennacl::ocl::context & context = viennacl::ocl::current_context();
		KeyType type_context(std::make_pair(feature_type,context.handle()));
		if(init_done[type_context]==false){
			std::cout << "Adding sources" << std::endl;
			
			//Compiles OpenCL Program for SVM.
			std::string pragma_option;
			pragma_option.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
			
			std::string sources(pragma_option);
			sources.append(dot_kernel_src(feature_type));
			sources.append( ocl_sources_map[std::make_pair(K_GAUSSIAN,feature_type)]);
			std::string prog_name = program_name(feature_type);
			
			std::cout << "Compiling sources" << std::endl;
			
			context.add_program(sources,prog_name);
			viennacl::ocl::program & prog = context.get_program(prog_name);
			prog.add_kernel("gaussian_kernel");
			prog.add_kernel("dot_kernel");
			init_done[type_context] = true;
		}
	}
	
	
	//General purpose kernels
      	
	general::SourcesMapType general::create_sources_map()
	{
		SourcesMapType m;
		m["apply"] = 
		#include "general/apply.cl"
				;
		return m;
	}

	std::string general::program_name(){
		return "svm_general";
	}
	
	void general::init(){
		static std::map<cl_context, bool> init_done;
		static SourcesMapType kernels_sources = create_sources_map();
		viennacl::ocl::context & context = viennacl::ocl::current_context();
		if(init_done[context.handle()]==false){

		//Compiles OpenCL Program for SVM.
		std::string pragma_option;
		pragma_option.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
		
		std::string sources(pragma_option);
		sources.append(kernels_sources["apply"]);
		
		std::string prog_name = program_name();
		context.add_program(sources,prog_name);
		viennacl::ocl::program & prog = context.get_program(prog_name);
		prog.add_kernel("apply");
		init_done[context.handle()] = true;
		}
	}
	
    }
    
  }
  
}