/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013-2014 Thoralf Klein
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013-2014 Thoralf Klein
 */

#include <shogun/labels/MultilabelLabels.h>

#include <string>
#include <iostream>
#include <fstream>
#include <limits>

using namespace shogun;

CMultilabelLabels::CMultilabelLabels ():CLabels ()
{
	m_num_labels = 0;
	m_num_classes = 1;
	init ();
}


CMultilabelLabels::CMultilabelLabels (mclass_t num_classes):CLabels ()
{
	m_num_labels = 0;
	m_num_classes = num_classes;
	init ();
}


CMultilabelLabels::CMultilabelLabels (int32_t num_labels, mclass_t num_classes):CLabels ()
{
	ASSERT (num_labels >= 0);
	ASSERT (num_classes > 0);
	m_num_labels = num_labels;
	m_num_classes = num_classes;
	init ();
}


CMultilabelLabels::~CMultilabelLabels ()
{
	delete[]m_labels;
}


void
CMultilabelLabels::init ()
{
	m_labels = new SGVector < mclass_t >[m_num_labels];
}


void
CMultilabelLabels::ensure_valid (const char *context)
{
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		if (!m_labels[label_j].is_sorted ())
		{
			printf ("m_labels[label_j=%d] not sorted: ", label_j);
			m_labels[label_j].display_vector ("");
		}
		ASSERT (m_labels[label_j].is_sorted ());
		int32_t c_len = m_labels[label_j].vlen;
		if (c_len > 0)
		{
			ASSERT (m_labels[label_j].vector[0] >= 0);
			ASSERT (m_labels[label_j].vector[c_len - 1] < get_num_classes ());
		}
	}
}


int32_t
CMultilabelLabels::get_num_labels () const const
{
	return m_num_labels;
}


mclass_t
CMultilabelLabels::get_num_classes () const const
{
	return m_num_classes;
}


void
CMultilabelLabels::set_labels (SGVector < mclass_t > *labels)
{
	for (int32_t label_j = 0; label_j < m_num_labels; label_j++)
	{
		m_labels[label_j] = labels[label_j];
	}
	ensure_valid ("set_labels()");
}


SGVector < int32_t > **CMultilabelLabels::get_class_labels () constconst
{
	SGVector < int32_t > **labels_list =
		SG_MALLOC (SGVector < int32_t > *, get_num_classes ());
	int32_t *
		num_label_idx = SG_MALLOC (int32_t, get_num_classes ());
	for (mclass_t class_i = 0; class_i < get_num_classes (); class_i++)
	{
		num_label_idx[class_i] = 0;
	}
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		for (int32_t c_pos = 0; c_pos < m_labels[label_j].vlen; c_pos++)
		{
			mclass_t
				class_i = m_labels[label_j][c_pos];
			ASSERT (class_i < get_num_classes ());
			num_label_idx[class_i]++;
		}
	}
	for (mclass_t class_i = 0; class_i < get_num_classes (); class_i++)
	{
		labels_list[class_i] =
			new SGVector < int32_t > (num_label_idx[class_i]);
	}
	SG_FREE (num_label_idx);
	int32_t *
		next_label_idx = SG_MALLOC (int32_t, get_num_classes ());
	for (mclass_t class_i = 0; class_i < get_num_classes (); class_i++)
	{
		next_label_idx[class_i] = 0;
	}
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		for (int32_t c_pos = 0; c_pos < m_labels[label_j].vlen; c_pos++)
		{
			// get class_i of current position
			mclass_t
				class_i = m_labels[label_j][c_pos];
			ASSERT (class_i < get_num_classes ());
			// next free element in m_classes[class_i]:
			int32_t
				l_pos = next_label_idx[class_i];
			ASSERT (l_pos < labels_list[class_i]->size ());
			next_label_idx[class_i]++;
			// finally, story label_j into class-column
			(*labels_list[class_i])[l_pos] = label_j;
		}
	}
	SG_FREE (next_label_idx);
	return labels_list;
}


SGVector < mclass_t > CMultilabelLabels::get_label (int32_t j)
{
	ASSERT (j < get_num_labels ());
	return m_labels[j];
}


template < class S, class D >
SGVector < D > CMultilabelLabels::to_dense (SGVector < S > *sparse,
int32_t dense_len, D d_true,
D d_false)
{
	SGVector < D > dense (dense_len);
	dense.set_const (d_false);
	for (int32_t i = 0; i < sparse->vlen; i++)
	{
		S
			index = (*sparse)[i];
		ASSERT (index < dense_len);
		dense[index] = d_true;
	}
	return dense;
}


template
SGVector < float64_t > CMultilabelLabels::to_dense <mclass_t, float64_t>
(SGVector < mclass_t > *, int32_t, float64_t, float64_t);

template
SGVector < int32_t > CMultilabelLabels::to_dense < mclass_t, int32_t > 
(SGVector < mclass_t > *, int32_t, int32_t, int32_t);

template
SGVector < float64_t > CMultilabelLabels::to_dense < int32_t, float64_t >
(SGVector < int32_t > *, int32_t, float64_t, float64_t);

void
CMultilabelLabels::set_label (int32_t j, SGVector < mclass_t > label)
{
	ASSERT (j < get_num_labels ());
	m_labels[j] = label;
}


void
CMultilabelLabels::set_class_labels (SGVector < int32_t > **labels_list)
{
	mclass_t *num_class_idx = SG_MALLOC (mclass_t, get_num_labels ());
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		num_class_idx[label_j] = 0;
	}
	for (mclass_t class_i = 0; class_i < get_num_classes (); class_i++)
	{
		for (int32_t l_pos = 0; l_pos < labels_list[class_i]->vlen; l_pos++)
		{
			int32_t label_j = (*labels_list[class_i])[l_pos];
			REQUIRE (label_j < get_num_labels (),
				"class_i=%d/%d :: label_j=%d/%d (l_pos=%d)\n",
				class_i, get_num_classes (), label_j, get_num_labels (),
				l_pos);
			ASSERT (label_j < get_num_labels ());
			num_class_idx[label_j]++;
		}
	}
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		m_labels[label_j].resize_vector (num_class_idx[label_j]);
	}
	SG_FREE (num_class_idx);
	mclass_t *next_class_idx = SG_MALLOC (mclass_t, get_num_labels ());
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		next_class_idx[label_j] = 0;
	}
	for (mclass_t class_i = 0; class_i < get_num_classes (); class_i++)
	{
		for (int32_t l_pos = 0; l_pos < labels_list[class_i]->vlen; l_pos++)
		{
			// get class_i of current position
			int32_t label_j = (*labels_list[class_i])[l_pos];
			ASSERT (label_j < get_num_labels ());
			// next free element in m_labels[label_j]:
			int32_t c_pos = next_class_idx[label_j];
			ASSERT (c_pos < m_labels[label_j].size ());
			next_class_idx[label_j]++;
			// finally, story label_j into class-column
			m_labels[label_j][c_pos] = class_i;
		}
	}
	SG_FREE (next_class_idx);
}


void
CMultilabelLabels::display () const const
{
	SGVector < int32_t > **labels_list = get_class_labels ();
	printf ("printing %d binary label vectors for %d multilabels:\n",
		get_num_classes (), get_num_labels ());
	for (int32_t class_i = 0; class_i < get_num_classes (); class_i++)
	{
		printf ("  yC_{class_i=%d}", class_i);
		SGVector < float64_t > dense =
			to_dense < int32_t, float64_t > (labels_list[class_i],
			get_num_labels (), +1, -1);
		dense.display_vector ("");
		delete labels_list[class_i];
	}
	SG_FREE (labels_list);
	printf ("printing %d binary class vectors for %d labels:\n",
		get_num_labels (), get_num_classes ());
	for (int32_t j = 0; j < get_num_labels (); j++)
	{
		printf ("  y_{j=%d}", j);
		SGVector < float64_t > dense =
			to_dense < mclass_t, float64_t > (&m_labels[j], get_num_classes (),
			+1, -1);
		dense.display_vector ("");
	}
	return;
}


void
CMultilabelLabels::save (const char *fname)
{
	FILE *fh = fopen (fname, "wb");
	for (int32_t label_j = 0; label_j < get_num_labels (); label_j++)
	{
		SGVector < mclass_t > yb = get_label (label_j);
		for (int32_t i_pos = 0; i_pos < yb.vlen; i_pos++)
		{
			fprintf (fh, "%d ", yb[i_pos]);
		}
		fprintf (fh, "\n");
	}
	fclose (fh);
}


void
CMultilabelLabels::load_info (const char *fname, int32_t & num_labels,
mclass_t & num_classes)
{
	std::ifstream labelfile (fname);
	std::string line;
	int32_t lineno = 0;
	mclass_t max_class_index = 0;
	while (std::getline (labelfile, line))
	{
		ASSERT (lineno < std::numeric_limits < int32_t >::max ());
		std::istringstream iss (line);
		std::string token;
		while (iss >> token)
		{
			mclass_t class_index;
			if ((std::istringstream (token) >> class_index).fail ())
			{
				SG_SERROR
					("INPUT ERROR (line %d): cannot cast token %s to integer\n",
					lineno + 1, token.c_str ());
				break;
			}
			ASSERT (class_index >= 0);
			ASSERT (class_index < std::numeric_limits < mclass_t >::max ());
			if (class_index > max_class_index)
			{
				max_class_index = class_index;
			}
		}
		lineno++;
	}
	labelfile.close ();
	max_class_index++;
	num_labels = lineno;
	num_classes = CMath::max (max_class_index, num_classes);
	// num_classes = CMath::max(max_class_index + 1, num_classes);
	// num_classes = max_class_index + 1 > num_classes ? max_class_index + 1 : num_classes;
	return;
}


CMultilabelLabels *
CMultilabelLabels::load (const char *fname)
{
	int32_t num_labels = 0;
	mclass_t num_classes = 0;
	CMultilabelLabels::load_info (fname, num_labels, num_classes);
	SG_SINFO
		("CMultilabelLabels::load(%s): found %d multilabels with %d classes\n",
		fname, num_labels, num_classes);
	mclass_t temp[num_classes];
	SGVector < mclass_t > *output_rows =
		SG_CALLOC (SGVector < mclass_t >, num_labels);
	std::ifstream labelfile (fname);
	std::string line;
	int32_t label_j = 0;
	while (std::getline (labelfile, line))
	{
		// std::cout << "input(line " << label_j << "): " << line << std::endl;
		ASSERT (label_j < num_labels);
		int32_t num_label_classes = 0;
		std::istringstream iss (line);
		std::string token;
		while (iss >> token)
		{
			mclass_t class_i;
			if ((std::istringstream (token) >> class_i).fail ())
			{
				SG_SERROR
					("INPUT ERROR (line %d): cannot cast token %s to integer\n",
					label_j + 1, token.c_str ());
				break;
			}
			ASSERT (class_i >= 0);
			ASSERT (class_i < num_classes);
			ASSERT (num_label_classes < num_classes);
			temp[num_label_classes] = class_i;
			num_label_classes++;
		}
		output_rows[label_j] =
			SGVector < mclass_t > (SGVector <
			mclass_t >::clone_vector (temp,
			num_label_classes),
			num_label_classes);
		label_j++;
	}
	ASSERT (label_j == num_labels);
	CMultilabelLabels *outputs =
		new CMultilabelLabels (num_labels, num_classes);
	outputs->set_labels (output_rows);
	SG_FREE (output_rows);
	return outputs;
}
