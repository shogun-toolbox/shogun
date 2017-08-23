/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/structure/SequenceLabels.h>

using namespace shogun;

CSequenceLabels::CSequenceLabels()
: CStructuredLabels()
{
}

CSequenceLabels::CSequenceLabels(index_t num_labels, index_t num_states)
    : CStructuredLabels(num_labels), m_num_states(num_states)
{
	init();
}

CSequenceLabels::CSequenceLabels(
    SGVector<index_t> labels, index_t label_length, index_t num_labels,
    index_t num_states)
    : CStructuredLabels(num_labels), m_num_states(num_states)
{
	REQUIRE(labels.vlen == label_length*num_labels, "The length of the labels must be "
			"equal to label_length times num_labels\n");
	init();

	for (index_t i = 0; i < labels.vlen; i += label_length)
	{
		add_vector_label(
		    SGVector<index_t>(
		        SGVector<index_t>::clone_vector(
		            labels.vector + i, label_length),
		        label_length));
	}
}

CSequenceLabels::~CSequenceLabels()
{
}

void CSequenceLabels::add_vector_label(SGVector<index_t> label)
{
	CStructuredLabels::add_label( new CSequence(label) );
}

void CSequenceLabels::init()
{
	SG_ADD(&m_num_states, "m_num_states", "Number of states", MS_NOT_AVAILABLE);
}
