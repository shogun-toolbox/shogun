function(find_meta_examples)
	FILE(GLOB_RECURSE META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/*.sg)
	
	# temporary hacks to exclude certain meta examples that have dependencies
	IF(NOT HAVE_NLOPT)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/gaussian_processes/gaussian_process_regression.sg)
	ENDIF()
	
	IF(NOT USE_GPL_SHOGUN)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/gaussian_processes/gaussian_process_regression.sg)
	ENDIF()
	
	IF(NOT HAVE_LAPACK)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/regression/linear_ridge_regression.sg)
	ENDIF()

	IF(NOT USE_SVMLIGHT)
		LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/regression/multiple_kernel_learning.sg)
	ENDIF()

	SET(META_EXAMPLES ${META_EXAMPLE_LISTINGS} PARENT_SCOPE)
endfunction()
