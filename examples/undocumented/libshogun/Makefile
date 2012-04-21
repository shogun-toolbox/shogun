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

VALGRIND_LOG = valgrind.log

TARGETS = basic_minimal \
		  classifier_libsvm \
		  classifier_minimal_svm \
		  classifier_mklmulticlass \
  		  classifier_conjugateindex \
		  classifier_gaussiannaivebayes \
		  classifier_multiclasslibsvm \
		  classifier_qda \
		  classifier_multiclasslinearmachine \
		  kernel_gaussian kernel_revlin \
		  library_dyn_int library_gc_array library_indirect_object \
		  library_hash parameter_set_from_parameters \
		  parameter_iterate_float64 parameter_iterate_sgobject \
		  parameter_modsel_parameters \
		  evaluation_cross_validation_classification \
		  evaluation_cross_validation_regression \
		  evaluation_cross_validation_locked_comparison \
		  modelselection_parameter_combination_test \
		  regression_gaussian_process \
		  modelselection_model_selection_parameters_test \
		  modelselection_parameter_tree \
		  modelselection_apply_parameter_tree \
		  modelselection_grid_search_linear features_subset_labels \
		  modelselection_grid_search_kernel \
		  modelselection_grid_search_string_kernel \
		  features_subset_stack \
		  features_subset_simple_features \
		  features_subset_sparse_features \
		  features_copy_subset_simple_features \
		  features_copy_subset_string_features \
		  features_copy_subset_sparse_features \
		  mathematics_confidence_intervals \
		  clustering_kmeans base_parameter_map \
		  base_load_file_parameters \
		  base_load_all_file_parameters \
		  base_map_parameters \
		  base_migration_type_conversion \
		  base_migration_dropping_and_new \
		  base_migration_multiple_dependencies \
		  splitting_stratified_crossvalidation \
		  splitting_standard_crossvalidation \
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
		  converter_stochasticproximityembedding \
		  serialization_basic_tests \
		  library_cover_tree \
		  kernel_machine_train_locked \
		  statistics \

all: $(TARGETS)

check-valgrind:
	for t in $(TARGETS); do echo $$t && valgrind --log-file=$(VALGRIND_LOG) ./$$t >/dev/null && cat $(VALGRIND_LOG) | grep SUMMARY && rm -f $(VALGRIND_LOG); done

print_targets:
	@echo $(TARGETS)

clean:
	rm -f $(TARGETS)
	rm -f $(VALGRIND_LOG)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && $(LIBRARY_PATH)="$(LIB_PATH):$$$(LIBRARY_PATH)" ./$$t >/dev/null; done

%: %.cpp
	$(CC) $(CXXFLAGS) $< $(LINKFLAGS) -o $@
