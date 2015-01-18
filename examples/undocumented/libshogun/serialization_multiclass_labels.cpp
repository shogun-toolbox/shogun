/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SerializableAsciiFile.h>

using namespace shogun;

void test()
{
	index_t n=10;
	index_t n_class=3;

	CMulticlassLabels* labels=new CMulticlassLabels();

	SGVector<float64_t> lab(n);
	for (index_t i=0; i<n; ++i)
		lab[i]=i%n_class;

	labels->set_labels(lab);

	labels->allocate_confidences_for(n_class);
	SGVector<float64_t> conf(n_class);
	for (index_t i=0; i<n_class; ++i)
		conf[i]=CMath::randn_double();

	for (index_t i=0; i<n; ++i)
		labels->set_multiclass_confidences(i, conf);

	/* create serialized copy */
	const char* filename="multiclass_labels.txt";
	CSerializableAsciiFile* file=new CSerializableAsciiFile(filename, 'w');
	labels->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file=new CSerializableAsciiFile(filename, 'r');
	CMulticlassLabels* labels_loaded=new CMulticlassLabels();
	labels_loaded->load_serializable(file);
	file->close();
	SG_UNREF(file);

	/* compare */
	labels->get_labels().display_vector("labels");
	labels_loaded->get_labels().display_vector("labels_loaded");

	for (index_t i=0; i<n_class; ++i)
	{
		labels->get_multiclass_confidences(i).display_vector("confidences");
		labels_loaded->get_multiclass_confidences(i).display_vector("confidences_loaded");
	}

	SG_UNREF(labels_loaded);
	SG_UNREF(labels);
}

int main()
{
	init_shogun_with_defaults();

//	sg_io->set_loglevel(MSG_DEBUG);

	test();

	exit_shogun();
	return 0;
}

