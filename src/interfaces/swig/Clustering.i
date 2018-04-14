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
%rename(DistanceMachine) CDistanceMachine;
%rename(Hierarchical) CHierarchical;
%rename(KMeans) CKMeans;
%rename(KMeansBase) CKMeansBase;
%rename(KMeansMiniBatch) CKMeansMiniBatch;
%rename(GMM) CGMM;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/machine/Machine.h>
%include <shogun/machine/DistanceMachine.h>
%include <shogun/clustering/KMeansBase.h> 
%include <shogun/clustering/KMeans.h>
%include <shogun/clustering/KMeansMiniBatch.h>
%include <shogun/clustering/Hierarchical.h>
%include <shogun/clustering/GMM.h>
