macro(read_config_header config_file)
	if(EXISTS ${config_file})
		FILE(READ ${config_file} SHOGUN_CONFIG)
	else()
		message(fatal_error "Could not locate config.h of shogun at: ${config_file}")
	endif()
endmacro()

function(is_set CONFIG_FILE FLAG)
	STRING(REGEX MATCH "\#define ${FLAG}" IS_SET_${FLAG} ${CONFIG_FILE})
	if (IS_SET_${FLAG})
		set(${FLAG} ON PARENT_SCOPE)
	else()
		set(${FLAG} OFF PARENT_SCOPE)
	endif()
endfunction()

function(find_meta_examples)
	read_config_header(${shogun_INCLUDE_DIR}/shogun/lib/config.h)
	is_set(${SHOGUN_CONFIG} HAVE_NLOPT)
	is_set(${SHOGUN_CONFIG} USE_GPL_SHOGUN)
	is_set(${SHOGUN_CONFIG} HAVE_LAPACK)
	is_set(${SHOGUN_CONFIG} USE_SVMLIGHT)

	FILE(GLOB_RECURSE META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/*.sg)

	# temporary hacks to exclude certain meta examples that have dependencies
	IF(NOT HAVE_NLOPT)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS
			${CMAKE_SOURCE_DIR}/examples/meta/src/gaussian_processes/gaussian_process_regression.sg)
	ENDIF()

	IF(NOT USE_GPL_SHOGUN)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS
			${CMAKE_SOURCE_DIR}/examples/meta/src/gaussian_processes/gaussian_process_regression.sg
			${CMAKE_SOURCE_DIR}/examples/meta/src/multiclass_classifier/multiclass_logisticregression.sg
			)
	ENDIF()

	IF(NOT HAVE_LAPACK)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS
			${CMAKE_SOURCE_DIR}/examples/meta/src/regression/linear_ridge_regression.sg
			${CMAKE_SOURCE_DIR}/examples/meta/src/clustering/gmm.sg
			${CMAKE_SOURCE_DIR}/examples/meta/src/distance/mahalanobis.sg
			)
	ENDIF()

	IF(NOT USE_SVMLIGHT)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS
			${CMAKE_SOURCE_DIR}/examples/meta/src/regression/multiple_kernel_learning.sg)
	ENDIF()

	SET(META_EXAMPLES ${META_EXAMPLE_LISTINGS} PARENT_SCOPE)
endfunction()
