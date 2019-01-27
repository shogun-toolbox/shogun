set(STAN_INCLUDE_DIR_STAN_MATH ${CMAKE_BINARY_DIR}/StanMath/src/StanMath)
set(STAN_INCLUDE_DIR_BOOST ${STAN_INCLUDE_DIR_STAN_MATH}/lib/boost_1.66.0)
set(STAN_INCLUDE_DIR_SUNDIALS ${STAN_INCLUDE_DIR_STAN_MATH}/lib/sundials_3.1.0/include)
set(STAN_INCLUDE_DIR_EIGEN ${STAN_INCLUDE_DIR_STAN_MATH}/lib/eigen_3.3.3)

include(ExternalProject)
ExternalProject_Add(
	StanMath
	PREFIX ${CMAKE_BINARY_DIR}/StanMath
	DOWNLOAD_DIR ${THIRD_PARTY_DIR}/StanMath
	URL https://github.com/stan-dev/math/archive/v2.18.1.tar.gz
	URL_MD5 0e7bdc294143b317c8e1ccee78d79fa0
	CONFIGURE_COMMAND ""
	BUILD_COMMAND ""
	INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${STAN_INCLUDE_DIR_STAN_MATH}/stan ${THIRD_PARTY_INCLUDE_DIR}/stan
		&& ${CMAKE_COMMAND} -E copy_directory ${STAN_INCLUDE_DIR_BOOST} ${THIRD_PARTY_INCLUDE_DIR}/stan_boost
		&& ${CMAKE_COMMAND} -E copy_directory ${STAN_INCLUDE_DIR_SUNDIALS} ${THIRD_PARTY_INCLUDE_DIR}/stan_sundials
		&& ${CMAKE_COMMAND} -E copy_directory ${STAN_INCLUDE_DIR_EIGEN} ${THIRD_PARTY_INCLUDE_DIR}/eigen
	LOG_DOWNLOAD ON
	)
add_dependencies(libshogun StanMath)
