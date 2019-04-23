/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Viktor Gal, Dawei Chen, Vladimir Perić, 
 *          Sergey Lisitsyn, Bjoern Esser
 */

#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>

#include <shogun/classifier/svm/OnlineLibLinear.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingSparseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

using namespace shogun;

int main(int argc, char* argv[])
{
    float64_t C = 1.0;
    char *train_file_name = (char*)"../data/train_sparsereal.light";
    char *test_file_name = (char*)"../data/test_sparsereal.light";
    char filename_tmp[] = "test_labels.XXXXXX";
#ifdef _WIN32
    int err = _mktemp_s(filename_tmp, strlen(filename_tmp)+1);
    ASSERT(err == 0);
#else
    int fd = mkstemp(filename_tmp);
    ASSERT(fd != -1);
    int retval = close(fd);
    ASSERT(retval != -1);
#endif
    char *test_labels_file_name = filename_tmp;

    if (argc > 4) {
        int32_t idx = 1;

        C = atof(argv[idx++]);
        train_file_name = argv[idx++];
        test_file_name = argv[idx++];
        test_labels_file_name = argv[idx++];

        ASSERT(idx <= argc);
    }

    fprintf(stderr, "*** training file %s with C %g\n", train_file_name, C);

    // Create an OnlineLiblinear object from the features. The first parameter is 'C'.
    auto svm = std::make_shared<OnlineLibLinear>(C);
    svm->set_bias_enabled(true);

    {
        Time train_time;
        train_time.start();

        // Create a StreamingAsciiFile from the training data
        auto train_file = std::make_shared<StreamingAsciiFile>(train_file_name);

        // The bool value is true if examples are labelled.
        // 1024 is a good standard value for the number of examples for the parser to hold at a time.
        auto train_features =
            std::make_shared<StreamingSparseFeatures < float32_t >> (train_file, true, 1024);

        svm->set_features(train_features);
        svm->train();

        train_file->close();

        train_time.stop();

        SGVector<float32_t> w_now = svm->get_w().clone();
        float32_t w_now_norm  = SGVector<float32_t>::twonorm(w_now.vector, w_now.vlen);

        uint64_t train_time_int = train_time.cur_time_diff();
        fprintf(stderr,
            "*** total training time: %llum%llus (or %.1f sec), #dim = %d, ||w|| = %f\n",
            train_time_int / 60, train_time_int % 60, train_time.cur_time_diff(),
            w_now.vlen, w_now_norm
        );
    }


    {
        Time test_time;
        test_time.start();

        // Now we want to test on holdout data
        auto test_file = std::make_shared<StreamingAsciiFile>(test_file_name);

        // Set second parameter to 'false' if the file contains unlabelled examples
        auto test_features =
            std::make_shared<StreamingSparseFeatures < float32_t >> (test_file, true, 1024);

        // Apply on all examples and return a BinaryLabels*
        auto test_binary_labels = svm->apply_binary(test_features);

        test_time.stop();
        uint64_t test_time_int = test_time.cur_time_diff();
        fprintf(stderr, "*** testing took %llum%llus (or %.1f sec)\n",
            test_time_int / 60, test_time_int % 60, test_time.cur_time_diff());


        // Writing labels for evaluation
        fprintf(stderr, "*** writing labels to file %s\n", test_labels_file_name);
        FILE* fh = fopen(test_labels_file_name, "wb");
        ASSERT(fh);

        for (int32_t j = 0; j < test_binary_labels->get_num_labels(); j++)
            fprintf(fh, "%d\n", test_binary_labels->get_int_label(j));

        fclose(fh);
        unlink(test_labels_file_name);
    }

    return 0;
}
