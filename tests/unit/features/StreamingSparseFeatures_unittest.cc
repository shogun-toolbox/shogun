/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Thoralf Klein, Giovanni De Toni, Bjoern Esser
 */

#ifndef _WIN32
#include <unistd.h>
#endif
#include <gtest/gtest.h>

#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/features/streaming/StreamingSparseFeatures.h>
#include <shogun/io/LibSVMFile.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>
#include "../utils/Utils.h"

using namespace shogun;

TEST(StreamingSparseFeaturesTest, parse_file)
{
  char fname[] = "StreamingSparseFeatures_parse_file.XXXXXX";
  generate_temp_filename(fname);

  int32_t seed = 100;
  int32_t max_num_entries=20;
  int32_t max_label_value=1;
  float64_t max_entry_value=1;

  int32_t num_vec=10;
  int32_t num_feat=0;

  std::mt19937_64 prng(seed);
  UniformIntDistribution<int32_t> uniform_int_dist;
  UniformRealDistribution<float64_t> uniform_real_dist;

  SGSparseVector<float64_t>* data=SG_MALLOC(SGSparseVector<float64_t>, num_vec);
  float64_t* labels=SG_MALLOC(float64_t, num_vec);
  for (int32_t i=0; i<num_vec; i++)
  {
    data[i]=SGSparseVector<float64_t>(uniform_int_dist(prng, {0, max_num_entries}));
    labels[i]=(float64_t) uniform_int_dist(prng, {-max_label_value, max_label_value});
    for (int32_t j=0; j<data[i].num_feat_entries; j++)
    {
      int32_t feat_index=(j+1)*2;
      if (feat_index>num_feat)
        num_feat=feat_index;

      data[i].features[j].feat_index=feat_index-1;
      data[i].features[j].entry=uniform_real_dist(prng, {0.0, max_entry_value});
    }
  }
  auto fout = std::make_shared<LibSVMFile>(fname, 'w');
  fout->set_sparse_matrix(data, num_feat, num_vec, labels);

  auto file = std::make_shared<StreamingAsciiFile>(fname);
  auto stream_features =
    std::make_shared<StreamingSparseFeatures<float64_t>>(file, true, 8);

  stream_features->start_parser();
  index_t i = 0;
  while (stream_features->get_next_example())
  {
      SGSparseVector<float64_t> v = stream_features->get_vector();
      EXPECT_EQ(data[i].num_feat_entries, v.num_feat_entries);

      for (index_t j = 0; j < data[i].num_feat_entries; j++)
      {
        EXPECT_EQ(data[i].features[j].feat_index, v.features[j].feat_index);
        EXPECT_DOUBLE_EQ(data[i].features[j].entry, v.features[j].entry);
      }

      stream_features->release_example();
      i++;
  }
  stream_features->end_parser();


  SG_FREE(data);
  SG_FREE(labels);

  std::remove(fname);
}
