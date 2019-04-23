/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias
 */

#include <shogun/structure/SequenceLabels.h>

using namespace shogun;

SequenceLabels::SequenceLabels()
: StructuredLabels()
{
}

SequenceLabels::SequenceLabels(int32_t num_labels, int32_t num_states)
: StructuredLabels(num_labels), m_num_states(num_states)
{
	init();
}

SequenceLabels::SequenceLabels(SGVector< int32_t > labels, int32_t label_length,
		int32_t num_labels, int32_t num_states)
: StructuredLabels(num_labels), m_num_states(num_states)
{
	REQUIRE(labels.vlen == label_length*num_labels, "The length of the labels must be "
			"equal to label_length times num_labels\n");
	init();

	for ( int32_t i = 0 ; i < labels.vlen ; i += label_length )
	{
		add_vector_label(SGVector< int32_t >(
			SGVector< int32_t >::clone_vector(labels.vector+i, label_length),
			label_length));
	}
}

SequenceLabels::~SequenceLabels()
{
}

void SequenceLabels::add_vector_label(SGVector< int32_t > label)
{
	StructuredLabels::add_label( std::make_shared<Sequence>(label) );
}

void SequenceLabels::init()
{
	SG_ADD(&m_num_states, "m_num_states", "Number of states");
}
