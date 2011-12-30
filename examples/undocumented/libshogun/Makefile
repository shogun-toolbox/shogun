LIB_PATH=/usr/lib
INC_PATH=/usr/include

INCLUDES=
LIBS=
LIBS_ADD=-lshogun

LINKFLAGS=
CXXFLAGS=-O3 -g

LIBRARY_PATH=LD_LIBRARY_PATH
LINKFLAGS+=$(LIBS) $(LIBS_ADD) -L$(LIB_PATH) 
CXXFLAGS+=-I$(INC_PATH) $(INCLUDES)
CC=c++

TARGETS =	  basic_minimal classifier_libsvm classifier_minimal_svm \
		  classifier_mklmulticlass kernel_gaussian kernel_revlin \
		  library_dyn_int library_gc_array library_indirect_object \
		  library_hash parameter_set_from_parameters \
		  parameter_iterate_float64 parameter_iterate_sgobject \
		  parameter_modsel_parameters \
		  modelselection_parameter_combination_test \
		  modelselection_model_selection_parameters_test \
		  modelselection_parameter_tree \
		  modelselection_apply_parameter_tree \
		  modelselection_grid_search_linear features_subset_labels \
		  modelselection_grid_search_kernel \
		  modelselection_grid_search_string_kernel \
		  features_subset_simple_features \
		  features_subset_sparse_features \
		  features_copy_subset_simple_features \
		  features_copy_subset_string_features \
		  features_copy_subset_sparse_features \
		  mathematics_confidence_intervals \
		  clustering_kmeans base_parameter_map \
		  base_load_file_parameter \
		  splitting_stratified_crossvalidation \
		  mathematics_arpack \
		  library_fibonacci_heap \
		  library_hashset \
		  mathematics_lapack \
		  converter_locallylinearembedding \
		  converter_localtangentspacealignment \
		  converter_hessianlocallylinearembedding \
		  converter_kernellocallylinearembedding \
		  converter_kernellocaltangentspacealignment \
		  converter_multidimensionalscaling \
		  converter_isomap \
		  converter_diffusionmaps \
		  converter_laplacianeigenmaps \
		  converter_neighborhoodpreservingembedding \
		  converter_linearlocaltangentspacealignment \
		  converter_localitypreservingprojections \
		  serialization_vector_matrix \

all: $(TARGETS)

print_targets:
	@echo $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && $(LIBRARY_PATH)="$(LIB_PATH):$$$(LIBRARY_PATH)" ./$$t >/dev/null; done

%: %.cpp
	$(CC) $(CXXFLAGS) $< $(LINKFLAGS) -o $@
