/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/io/SerializableAsciiFile.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

const int32_t num_vectors=6;
const int32_t dim_features_save=6;
const char* filename="test.txt";

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	/* create feature data matrix with random data */
	SGMatrix<int32_t> data(dim_features_save, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		for (index_t j=0; j<dim_features_save; ++j)
			data.matrix[i*dim_features_save+j]=CMath::random(-5, 5);
	}

	/* create simple features */
	CSimpleFeatures<int32_t>* features_save=new CSimpleFeatures<int32_t> (data);
	SG_REF(features_save);

	/* serialize */
	CSerializableFile* file;
	file=new CSerializableAsciiFile(filename, 'w');
	features_save->save_serializable(file);
	file->close();
	SG_UNREF(file);

	/* deserialize */
	file=new CSerializableAsciiFile(filename, 'r');
	CSimpleFeatures<int32_t>* features_load=new CSimpleFeatures<int32_t> ();
	features_load->load_serializable(file);
	file->close();
	SG_UNREF(file);

	/* test deserialization */
	for (index_t i=0; i<features_load->get_num_features(); ++i)
	{
		SGVector<int32_t> vec1=features_load->get_feature_vector(i);
		SGVector<int32_t> vec2=features_save->get_feature_vector(i);

		ASSERT(vec1.vlen=vec2.vlen);
		for (index_t j=0; j<vec1.vlen; ++j)
			ASSERT(vec1.vector[j]==vec2.vector[j]);

		features_load->free_feature_vector(vec1, i);
		features_save->free_feature_vector(vec2, i);
	}

	/* cleanup */
	SG_UNREF(features_save);
	SG_UNREF(features_load);

	exit_shogun();

	return 0;
}

