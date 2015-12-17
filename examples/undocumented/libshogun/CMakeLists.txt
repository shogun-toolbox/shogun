INCLUDE_DIRECTORIES(${INCLUDES})
if(SYSTEM_INCLUDES)
	INCLUDE_DIRECTORIES(SYSTEM ${SYSTEM_INCLUDES})
endif()

FILE(GLOB EXAMPLES_CPP RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

IF (NOT USE_MOSEK)
	LIST(REMOVE_ITEM EXAMPLES_CPP "so_fg_model.cpp" "so_multiclass.cpp" "so_multilabel.cpp")
ENDIF ()

LIST(REMOVE_ITEM EXAMPLES_CPP
	"classifier_bagging_liblinear.cpp"
	"classifier_larank.cpp"
	"classifier_latent_svm.cpp"
	"classifier_multiclass_relaxedtree.cpp"
	"classifier_multiclass_shareboost.cpp"
	"classifier_nearest_centroid.cpp"
	"classifier_newtontest.cpp"
	"evaluation_cross_validation_knn.cpp"
	"hashed_features_example.cpp"
	"kernel_custom_kernel.cpp"
	"library_dynarray.cpp"
	"metric_lmnnn.cpp"
	"minibatchKMeans.cpp"
	"modelselection_grid_search_krr.cpp"
	"optimization_lbfgs.cpp"
	"preprocessor_fisherlda.cpp"
	"preprocessor_randomfouriergauss.cpp"
	"random_conditional_probability_tree.cpp"
	"random_fourier_features.cpp"
	"serialization_file_formats.cpp"
	"so_hmsvm_mosek_simple.cpp"
	"splitting_LOO_crossvalidation.cpp"
	"streaming_onlinesvmsgd.cpp"
	"streaming_stringfeatures.cpp"
	"streaming_vowpalwabbit.cpp"
	"streaming_vwfeatures.cpp"
	"structure_discrete_hmsvm_bmrm.cpp"
	"structure_hmsvm_mosek.cpp"
	"structure_plif_hmsvm_bmrm.cpp"
)

FOREACH(EXAMPLE_CPP ${EXAMPLES_CPP})
	STRING(REGEX REPLACE ".cpp\$" "" EXAMPLE "${EXAMPLE_CPP}")

	add_executable(${EXAMPLE} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_CPP})
	target_link_libraries(${EXAMPLE} shogun ${SANITIZER_LIBRARY})
	IF(SANITIZER_FLAGS)
		set_target_properties(${EXAMPLE} PROPERTIES COMPILE_FLAGS ${SANITIZER_FLAGS})
	ENDIF()
	add_test(libshogun-${EXAMPLE} ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE})
	set_property(TEST libshogun-${EXAMPLE} PROPERTY
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
	LIST(APPEND INSTALL_EXAMPLES ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE})

	# Add examples to the dependencies of modular interfaces to make sure
	# nothing will infer with them being build single-threaded.
	IF(SWIG_SINGLE_THREADED)
		FOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
			ADD_DEPENDENCIES(${SG_MODULAR_INTERFACE_TARGET} ${EXAMPLE})
		ENDFOREACH(SG_MODULAR_INTERFACE_TARGET ${SG_MODULAR_INTERFACE_TARGETS})
	ENDIF(SWIG_SINGLE_THREADED)

ENDFOREACH()

INSTALL(PROGRAMS ${INSTALL_EXAMPLES}
	DESTINATION share/shogun/examples/libshogun
	COMPONENT libshogun-examples)
