LIB_PATH=/usr/lib
INC_PATH=/usr/include

INCLUDES=
LIBS=-lshogun -lblas

LINKFLAGS=
CXXFLAGS=-O3 -g

LIBRARY_PATH=LD_LIBRARY_PATH
LINKFLAGS+=$(LIBS) -L$(LIB_PATH) 
CXXFLAGS+=-I$(INC_PATH) $(INCLUDES)
CC=c++

TARGETS = basic_minimal classifier_libsvm classifier_minimal_svm \
		  classifier_mklmulticlass kernel_gaussian kernel_revlin \
		  library_dyn_int library_gc_array library_indirect_object \
		  library_hash parameter_set_from_parameters \
		  parameter_iterate_float64 parameter_iterate_sgobject \
		  modelselection_parameter_combination_test \
		  modelselection_model_selection_parameters_test \
		  modelselection_parameter_tree \
		  modelselection_apply_parameter_tree \
		  modelselection_grid_search_linear features_subset_labels \
		  modelselection_grid_search_kernel \
		  modelselection_grid_search_string_kernel \
		  features_subset_simple_features \
		  features_subset_sparse_features \
		  mathematics_confidence_intervals \
		  features_copy_subset_simple_features \
		  features_copy_subset_string_features \
		  clustering_kmeans


all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS): $(foreach t, $(TARGET), $t.cpp)

check-examples: $(TARGETS)
	@for t in $(TARGETS) ; do echo $$t && $(LIBRARY_PATH)="$(LIB_PATH):$$$(LIBRARY_PATH)" ./$$t >/dev/null; done

%: %.cpp
	$(CC) $(CXXFLAGS) $< $(LINKFLAGS) -o $@
