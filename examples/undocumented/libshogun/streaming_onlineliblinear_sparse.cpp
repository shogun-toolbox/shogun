/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein
 * Copyright (C) 2013 Zuse-Institute-Berlin
 *
 * This example demonstrates use of the online learning with
 * OnlineLibLinear using sparse streaming features.  This example
 * also parses command line options: Can be used as stand-alone
 * program to do binary classifications on user-provided inputs.
 */

#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>

#include <shogun/classifier/svm/OnlineLibLinear.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingSparseFeatures.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

int main(int argc, char* argv[])
{
    init_shogun_with_defaults();

    float64_t C = 1.0;
    char *train_file_name = (char*)"../data/train_sparsereal.light";
    char *test_file_name = (char*)"../data/test_sparsereal.light";
    char filename_tmp[] = "test_sparsereal.light.labels.XXXXXX";
    int fd = mkstemp(filename_tmp);
    ASSERT(fd != -1);
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
    COnlineLibLinear *svm = new COnlineLibLinear(C);
    svm->set_bias_enabled(true);

    {
        CTime train_time;
        train_time.start();

        // Create a StreamingAsciiFile from the training data
        CStreamingAsciiFile *train_file = new CStreamingAsciiFile(train_file_name);
        SG_REF(train_file);

        // The bool value is true if examples are labelled.
        // 1024 is a good standard value for the number of examples for the parser to hold at a time.
        CStreamingSparseFeatures < float32_t > *train_features =
            new CStreamingSparseFeatures < float32_t > (train_file, true, 1024);
        SG_REF(train_features);

        svm->set_features(train_features);
        svm->train();

        train_file->close();
        SG_UNREF(train_file);
        SG_UNREF(train_features);

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
        CTime test_time;
        test_time.start();

        // Now we want to test on holdout data
        CStreamingAsciiFile *test_file = new CStreamingAsciiFile(test_file_name);
        SG_REF(test_file);

        // Set second parameter to 'false' if the file contains unlabelled examples
        CStreamingSparseFeatures < float32_t > *test_features =
            new CStreamingSparseFeatures < float32_t > (test_file, true, 1024);
        SG_REF(test_features);

        // Apply on all examples and return a CBinaryLabels*
        CBinaryLabels *test_binary_labels = svm->apply_binary(test_features);
        SG_REF(test_binary_labels);

        test_time.stop();
        uint64_t test_time_int = test_time.cur_time_diff();
        fprintf(stderr, "*** testing took %llum%llus (or %.1f sec)\n",
            test_time_int / 60, test_time_int % 60, test_time.cur_time_diff());

        SG_UNREF(test_features);
        SG_UNREF(test_file);

        // Writing labels for evaluation
        fprintf(stderr, "*** writing labels to file %s\n", test_labels_file_name);
        FILE* fh = fopen(test_labels_file_name, "wb");
        ASSERT(fh);

        for (int32_t j = 0; j < test_binary_labels->get_num_labels(); j++)
            fprintf(fh, "%d\n", test_binary_labels->get_int_label(j));

        fclose(fh);
        SG_UNREF(test_binary_labels);
        unlink(test_labels_file_name);
    }

    SG_UNREF(svm);
    exit_shogun();

    return 0;
}
