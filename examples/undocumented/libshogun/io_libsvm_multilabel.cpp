/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jiaolong Xu, Bjoern Esser
 */
#include <shogun/io/LibSVMFile.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/base/DynArray.h>
#include <shogun/base/ShogunEnv.h>

using namespace shogun;

#define SHOW_DATA

/* file data */
const char fname_svm_multilabel[] = "../../../../data/multilabel/yeast_test.svm";

void test_libsvmfile_multilabel(const char* fname)
{
	FILE* pfile = fopen(fname, "r");

    if (pfile == NULL)
	{
		SG_SPRINT("Unable to open file: %s\n", fname);
		return;
	}

    fclose(pfile);

	/* sparse data from matrix*/
	auto svmfile = std::make_shared<LibSVMFile>(fname);

	SGSparseVector<float64_t>* feats;
	SGVector<float64_t>* labels;
	int32_t dim_feat;
	int32_t num_samples;
	int32_t num_classes;

	svmfile->get_sparse_matrix(feats, dim_feat, num_samples, labels, num_classes);

#ifdef SHOW_DATA
	// Display the labels
	for (int32_t i = 0; i < num_samples; i++)
	{
		labels[i].display_vector();
	}
#endif

	SG_SPRINT("Number of the samples: %d\n", num_samples);
	SG_SPRINT("Dimention of the feature: %d\n", dim_feat);
	SG_SPRINT("Number of classes: %d\n", num_classes);

	SG_FREE(feats);
	SG_FREE(labels);
}

int main(int argc, char ** argv)
{
	env()->io()->set_loglevel(MSG_DEBUG);

	test_libsvmfile_multilabel(fname_svm_multilabel);

	return 0;
}
