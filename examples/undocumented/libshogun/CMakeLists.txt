FILE(GLOB EXAMPLES_CPP RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

IF (NOT USE_MOSEK)
	LIST(REMOVE_ITEM EXAMPLES_CPP "so_fg_model.cpp" "so_multiclass.cpp" "so_multilabel.cpp")
ENDIF ()

IF (NOT USE_GPL_SHOGUN)
	LIST(REMOVE_ITEM EXAMPLES_CPP
	"classifier_svmlight_string_features_precomputed_kernel.cpp"
	"classifier_larank.cpp"
	)
ENDIF()

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
	"optimization_lbfgs.cpp"
	"preprocessor_fisherlda.cpp"
	"preprocessor_randomfouriergauss.cpp"
	"random_conditional_probability_tree.cpp"
	"random_fourier_features.cpp"
	"serialization_file_formats.cpp"
	"so_fg_multilabel.cpp"
	"so_hmsvm_mosek_simple.cpp"
	"splitting_LOO_crossvalidation.cpp"
	"streaming_onlinesvmsgd.cpp"
	"streaming_vowpalwabbit.cpp"
	"structure_discrete_hmsvm_bmrm.cpp"
	"structure_hmsvm_mosek.cpp"
	"structure_plif_hmsvm_bmrm.cpp"
)

FOREACH(EXAMPLE_CPP ${EXAMPLES_CPP})
	STRING(REGEX REPLACE ".cpp\$" "" EXAMPLE "${EXAMPLE_CPP}")
    AddLibShogunExample(${EXAMPLE_CPP})
    LIST(APPEND EXAMPLE_TARGETS ${EXAMPLE})
ENDFOREACH()
