#include "dot_kernels.h"
#include "general.h"
#include <shogun/opencl/viennacl_compatibility.h>

namespace shogun{
  
  namespace ocl{
    	
    namespace svm{
	    		
	dot_kernels::SourcesMapType dot_kernels::create_sources_map()
	{
		SourcesMapType m;
		m[K_GAUSSIAN] = 
			#include "dot_kernels/gaussian_kernel.cl"
		;
		
		return m;
	}
	
		
	std::string dot_kernels::program_name(){
		return "svm_dot";
	}

	std::string dot_kernels::dot_kernel_name(EFeatureType feature_type){
		if(feature_type==F_SHORTREAL){
			return "dot_kernel_float";
		}
		if(feature_type==F_DREAL){
			return "dot_kernel_double";
		}
		return "";
	}
	
	void dot_kernels::init(){
		std::cout << "Init" << std::endl;
		static std::map<cl_context, bool> init_done;	
		static SourcesMapType ocl_sources_map = create_sources_map();
		viennacl::ocl::context & context = viennacl::ocl::current_context();
		if(init_done[context.handle()]==false){
			std::cout << "Adding sources" << std::endl;
			
			//Compiles OpenCL Program for SVM.
			std::string pragma_option;
			pragma_option.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
			
			std::string sources(pragma_option);
			
			std::string dot_kernel_float_src = 
				#include "dot_kernels/float/dot_kernel.cl"
			;
			std::string dot_kernel_double_src = 
				#include "dot_kernels/double/dot_kernel.cl"
			;
			
			sources.append(dot_kernel_float_src);
			sources.append(dot_kernel_double_src);
			sources.append(ocl_sources_map[K_GAUSSIAN]);
			std::string prog_name = program_name();
			
			std::cout << "Compiling sources" << std::endl;
			
			context.add_program(sources,prog_name);
			viennacl::ocl::program & prog = context.get_program(prog_name);
			prog.add_kernel("dot_kernel_float");
			prog.add_kernel("dot_kernel_double");
			prog.add_kernel("gaussian_kernel");
			init_done[context.handle()] = true;
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