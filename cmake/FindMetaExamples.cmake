function(find_meta_examples)
	FILE(GLOB_RECURSE META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/*.sg)
	SET(META_EXAMPLES ${META_EXAMPLE_LISTINGS} PARENT_SCOPE)
	
	# temporary hack to exclude certain meta examples that have dependencies
	IF(NOT HAVE_NLOPT)
		LIST(REMOVE_ITEM META_EXAMPLES ${CMAKE_SOURCE_DIR}/examples/meta/src/gaussian_processes/gaussian_process_regression.sg)
	ENDIF()
endfunction()
