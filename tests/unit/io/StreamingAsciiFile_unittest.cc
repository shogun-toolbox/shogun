/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Thoralf Klein <thoralf.klein@zib.de>
 */

#include <gtest/gtest.h>

#include <shogun/base/init.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingSparseFeatures.h>

using namespace shogun;

TEST(StreamingAsciiFile, DISABLED_parse_file)
{
  char *fname = (char *)"data/toy/train_sparsereal.light";

  CStreamingAsciiFile *file = new CStreamingAsciiFile(fname);
  CStreamingSparseFeatures<float64_t> *stream_features =
    new CStreamingSparseFeatures<float64_t>(file, true, 8);

  stream_features->start_parser();
  while (stream_features->get_next_example())
  {
      // stream_features->get_vector();
      stream_features->release_example();
  }
  stream_features->end_parser();

  SG_UNREF(stream_features);
  SG_UNREF(file);
}
