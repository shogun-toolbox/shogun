/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_radi(self) -> numpy 1dim array of float") get_radi;
%feature("autodoc", "get_centers(self) -> numpy 2dim array of float") get_centers;
%feature("autodoc", "get_merge_distance(self) -> [] of float") get_merge_distance;
%feature("autodoc", "get_pairs(self) -> [] of float") get_pairs;
#endif

/* Remove C Prefix */
%shared_ptr(shogun::DistanceMachine)
SHARED_RANDOM_INTERFACE(shogun::DistanceMachine)
%shared_ptr(shogun::Hierarchical)
%shared_ptr(shogun::KMeans)
%shared_ptr(shogun::KMeansBase)
%shared_ptr(shogun::KMeansMiniBatch)
%shared_ptr(shogun::GMM)

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/Machine.h>
%include <shogun/machine/DistanceMachine.h>
RANDOM_INTERFACE(DistanceMachine)
%include <shogun/clustering/KMeansBase.h> 
%include <shogun/clustering/KMeans.h>
%include <shogun/clustering/KMeansMiniBatch.h>
%include <shogun/clustering/Hierarchical.h>
%include <shogun/clustering/GMM.h>
