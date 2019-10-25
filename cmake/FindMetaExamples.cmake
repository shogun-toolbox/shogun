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

function(get_excluded_meta_examples)
    read_config_header(${shogun_INCLUDE_DIR}/shogun/lib/config.h)
    is_set(${SHOGUN_CONFIG} HAVE_NLOPT)
    is_set(${SHOGUN_CONFIG} USE_GPL_SHOGUN)
    is_set(${SHOGUN_CONFIG} HAVE_LAPACK)
    is_set(${SHOGUN_CONFIG} USE_SVMLIGHT)

    IF(NOT HAVE_NLOPT)
        LIST(APPEND EXCLUDED_META_EXAMPLES
            gaussian_process/regression.sg
            )
    ENDIF()

	IF(NOT USE_GPL_SHOGUN)
		LIST(APPEND EXCLUDED_META_EXAMPLES
			multiclass/logistic_regression.sg
      statistical_testing/linear_time_maximum_mean_discrepancy.sg
      statistical_testing/quadratic_time_maximum_mean_discrepancy.sg
      gaussian_process/classifier.sg
      gaussian_process/regression.sg
      binary/svmlin.sg
      binary/svmsgd.sg
      )
	ENDIF()

	IF(NOT HAVE_LAPACK)
		LIST(APPEND EXCLUDED_META_EXAMPLES
			)
	ENDIF()

	IF(NOT USE_SVMLIGHT)
		LIST(APPEND EXCLUDED_META_EXAMPLES
			regression/multiple_kernel_learning.sg
			binary/multiple_kernel_learning.sg
			binary/svmlight.sg
			binary/svmlight_batch_linadd.sg
			regression/svrlight.sg
		)
	ENDIF()
  
  # TODO: remove me when model selection is fixed
  LIST(APPEND EXCLUDED_META_EXAMPLES
    gaussian_process/regression.sg)

  # remove double entries to avoid errors due to "double" removing
  LIST(LENGTH EXCLUDED_META_EXAMPLES NUM_EXCLUDED)
  IF(NUM_EXCLUDED GREATER 0)
    LIST(REMOVE_DUPLICATES EXCLUDED_META_EXAMPLES)
  ENDIF()
  
  SET(EXCLUDED_META_EXAMPLES ${EXCLUDED_META_EXAMPLES} PARENT_SCOPE)

endfunction()

# Remove meta example that cannot be built because of missing dependencies
function(find_meta_examples)

	FILE(GLOB_RECURSE META_EXAMPLE_LISTINGS ${CMAKE_SOURCE_DIR}/examples/meta/src/*.sg)
    get_excluded_meta_examples()

    FOREACH(META_EXAMPLE ${EXCLUDED_META_EXAMPLES})
        SET(EXCLUDED_EXAMPLE_FNAME ${CMAKE_SOURCE_DIR}/examples/meta/src/${META_EXAMPLE})
        LIST(FIND META_EXAMPLE_LISTINGS ${EXCLUDED_EXAMPLE_FNAME} HAS_EXAMPLE)
 
        IF(HAS_EXAMPLE EQUAL -1)
            MESSAGE(FATAL_ERROR "Guarded meta example ${META_EXAMPLE} does not exist.")
        ENDIF()
        LIST(REMOVE_ITEM META_EXAMPLE_LISTINGS ${EXCLUDED_EXAMPLE_FNAME})
    ENDFOREACH()

	SET(META_EXAMPLES ${META_EXAMPLE_LISTINGS} PARENT_SCOPE)
endfunction()

# Get the cookbook pages we want to exclude from build
function(find_excluded_cookbook_pages)

    get_excluded_meta_examples()

    FOREACH(META_EXAMPLE ${EXCLUDED_META_EXAMPLES})
        # This is made since some meta example does not have
        # a cookbook page.
        STRING(REPLACE ".sg" ".rst" META_EXAMPLE ${META_EXAMPLE})
        IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/source/examples/${META_EXAMPLE})
            LIST(APPEND EXCLUDED_COOKBOOK_PAGES examples/${META_EXAMPLE})
        ENDIF()
    ENDFOREACH()

    # Check to ensure we have cookbook pages to exclude
    IF(EXCLUDED_COOKBOOK_PAGES)
        # Generate a string with all the meta examples separated by commas.
        # This is made since Sphinx's exclude_patterns option wants
        # the list's items separated by commas, but cmake's lists use
        # semicolons instead.
        # See: https://cmake.org/cmake/help/v3.3/command/list.html
        # See: http://www.sphinx-doc.org/en/stable/invocation.html#id2
        SET(TEMP "${EXCLUDED_COOKBOOK_PAGES}")
        STRING(REPLACE ".rst" ".rst," TEMP ${TEMP})
        string(REGEX REPLACE ".rst,$" ".rst" TEMP ${TEMP})

        SET(EXCLUDED_COOKBOOK_PAGES ${TEMP} PARENT_SCOPE)
    ELSE()
        SET(EXCLUDED_COOKBOOK_PAGES "" PARENT_SCOPE)
    ENDIF()


endfunction()
