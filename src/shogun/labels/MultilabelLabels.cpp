/*
 * Copyright (C) 2013 Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2013-2014 Thoralf Klein
 * Written (W) 2013-2014 Thoralf Klein
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/labels/MultilabelLabels.h>

using namespace shogun;

CMultilabelLabels::CMultilabelLabels()
	: CLabels()
{
	init(0, 1);
}


CMultilabelLabels::CMultilabelLabels(int16_t num_classes)
	: CLabels()
{
	init(0, num_classes);
}


CMultilabelLabels::CMultilabelLabels(int32_t num_labels, int16_t num_classes)
	: CLabels()
{
	init(num_labels, num_classes);
}


CMultilabelLabels::~CMultilabelLabels()
{
	delete[] m_labels;
}


void
CMultilabelLabels::init(int32_t num_labels, int16_t num_classes)
{
	REQUIRE(num_labels >= 0, "num_labels=%d should be >= 0", num_labels);
	REQUIRE(num_classes > 0, "num_classes=%d should be > 0", num_classes);

	// This one does consider the contained labels, so its simply BROKEN
	// Can be disabled as 
	SG_ADD(&m_num_labels, "m_num_labels", "number of labels", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_classes, "m_num_classes", "number of classes", MS_NOT_AVAILABLE);
	// SG_ADD((CSGObject**) &m_labels, "m_labels", "The labels", MS_NOT_AVAILABLE);


	// Can only be enabled after this issue has been solved:
	// https://github.com/shogun-toolbox/shogun/issues/1972
/*	this->m_parameters->add(&m_num_labels, "m_num_labels",
	                        "Number of labels.");
	this->m_parameters->add(&m_num_classes, "m_num_classes",
	                        "Number of classes.");
	this->m_parameters->add_vector(&m_labels, &m_num_labels, "labels_array",
	                               "The label vectors for all (num_labels) outputs.");
*/

	m_num_labels = num_labels;
	m_num_classes = num_classes;
	m_labels = new SGVector <int16_t>[m_num_labels];
}


void
CMultilabelLabels::ensure_valid(const char * context)
{
	for (int32_t label_j = 0; label_j < get_num_labels(); label_j++)
	{
		if (sg_io->get_loglevel() == MSG_DEBUG && !m_labels[label_j].is_sorted())
		{
			SG_PRINT("m_labels[label_j=%d] not sorted: ", label_j);
			m_labels[label_j].display_vector("");
		}

		REQUIRE(m_labels[label_j].is_sorted(),
		        "labels[%d] are not sorted!", label_j);

		int32_t c_len = m_labels[label_j].vlen;
		if (c_len <= 0)
		{
			continue;
		}

		REQUIRE(m_labels[label_j].vector[0] >= 0,
		        "first label labels[%d]=%d should be >= 0!",
		        label_j, m_labels[label_j].vector[0]);
		REQUIRE(m_labels[label_j].vector[c_len - 1] < get_num_classes(),
		        "last label labels[%d]=%d should be < num_classes == %d!",
		        label_j, m_labels[label_j].vector[0], get_num_classes());
	}
}


int32_t
CMultilabelLabels::get_num_labels() const
{
	return m_num_labels;
}


int16_t
CMultilabelLabels::get_num_classes() const
{
	return m_num_classes;
}


void
CMultilabelLabels::set_labels(SGVector <int16_t> * labels)
{
	for (int32_t label_j = 0; label_j < m_num_labels; label_j++)
	{
		m_labels[label_j] = labels[label_j];
	}
	ensure_valid("set_labels()");
}


SGVector <int32_t> ** CMultilabelLabels::get_class_labels() const
{
	SGVector <int32_t> ** labels_list =
	        SG_MALLOC(SGVector <int32_t> *, get_num_classes());
	int32_t * num_label_idx =
	        SG_MALLOC(int32_t, get_num_classes());

	for (int16_t  class_i = 0; class_i < get_num_classes(); class_i++)
	{
		num_label_idx[class_i] = 0;
	}

	for (int32_t label_j = 0; label_j < get_num_labels(); label_j++)
	{
		for (int32_t c_pos = 0; c_pos < m_labels[label_j].vlen; c_pos++)
		{
			int16_t class_i = m_labels[label_j][c_pos];
			REQUIRE(class_i < get_num_classes(),
			        "class_i exceeded number of classes");
			num_label_idx[class_i]++;
		}
	}

	for (int16_t  class_i = 0; class_i < get_num_classes(); class_i++)
	{
		labels_list[class_i] =
		        new SGVector <int32_t> (num_label_idx[class_i]);
	}
	SG_FREE(num_label_idx);

	int32_t * next_label_idx = SG_MALLOC(int32_t, get_num_classes());
	for (int16_t  class_i = 0; class_i < get_num_classes(); class_i++)
	{
		next_label_idx[class_i] = 0;
	}

	for (int32_t label_j = 0; label_j < get_num_labels(); label_j++)
	{
		for (int32_t c_pos = 0; c_pos < m_labels[label_j].vlen; c_pos++)
		{
			// get class_i of current position
			int16_t class_i = m_labels[label_j][c_pos];
			REQUIRE(class_i < get_num_classes(),
			        "class_i exceeded number of classes");
			// next free element in m_classes[class_i]:
			int32_t l_pos = next_label_idx[class_i];
			REQUIRE(l_pos < labels_list[class_i]->size(),
			        "l_pos exceeded length of label list");
			next_label_idx[class_i]++;
			// finally, story label_j into class-column
			(*labels_list[class_i])[l_pos] = label_j;
		}
	}

	SG_FREE(next_label_idx);
	return labels_list;
}


SGVector <int16_t> CMultilabelLabels::get_label(int32_t j)
{
	REQUIRE(j < get_num_labels(),
	        "label index j=%d should be within [%d,%d[",
	        j, 0, get_num_labels());
	return m_labels[j];
}


template <class S, class D>
SGVector <D> CMultilabelLabels::to_dense
(SGVector <S> * sparse, int32_t dense_len, D d_true, D d_false)
{
	SGVector <D> dense(dense_len);
	dense.set_const(d_false);
	for (int32_t i = 0; i < sparse->vlen; i++)
	{
		S index = (*sparse)[i];
		REQUIRE(index < dense_len,
		        "class index exceeded length of dense vector");
		dense[index] = d_true;
	}
	return dense;
}


template
SGVector <float64_t> CMultilabelLabels::to_dense <int16_t, float64_t>
(SGVector <int16_t> *, int32_t, float64_t, float64_t);

template
SGVector <int32_t> CMultilabelLabels::to_dense <int16_t, int32_t>
(SGVector <int16_t> *, int32_t, int32_t, int32_t);

template
SGVector <float64_t> CMultilabelLabels::to_dense <int32_t, float64_t>
(SGVector <int32_t> *, int32_t, float64_t, float64_t);

void
CMultilabelLabels::set_label(int32_t j, SGVector <int16_t> label)
{
	REQUIRE(j < get_num_labels(),
	        "label index j=%d should be within [%d,%d[",
	        j, 0, get_num_labels());
	m_labels[j] = label;
}


void
CMultilabelLabels::set_class_labels(SGVector <int32_t> ** labels_list)
{
	int16_t * num_class_idx = SG_MALLOC(int16_t , get_num_labels());
	for (int32_t label_j = 0; label_j < get_num_labels(); label_j++)
	{
		num_class_idx[label_j] = 0;
	}

	for (int16_t class_i = 0; class_i < get_num_classes(); class_i++)
	{
		for (int32_t l_pos = 0; l_pos < labels_list[class_i]->vlen; l_pos++)
		{
			int32_t label_j = (*labels_list[class_i])[l_pos];
			REQUIRE(label_j < get_num_labels(),
			        "class_i=%d/%d :: label_j=%d/%d (l_pos=%d)\n",
			        class_i, get_num_classes(), label_j, get_num_labels(),
			        l_pos);
			num_class_idx[label_j]++;
		}
	}

	for (int32_t label_j = 0; label_j < get_num_labels(); label_j++)
	{
		m_labels[label_j].resize_vector(num_class_idx[label_j]);
	}
	SG_FREE(num_class_idx);

	int16_t * next_class_idx = SG_MALLOC(int16_t , get_num_labels());
	for (int32_t label_j = 0; label_j < get_num_labels(); label_j++)
	{
		next_class_idx[label_j] = 0;
	}

	for (int16_t class_i = 0; class_i < get_num_classes(); class_i++)
	{
		for (int32_t l_pos = 0; l_pos < labels_list[class_i]->vlen; l_pos++)
		{
			// get class_i of current position
			int32_t label_j = (*labels_list[class_i])[l_pos];
			REQUIRE(label_j < get_num_labels(),
			        "class_i=%d/%d :: label_j=%d/%d (l_pos=%d)\n",
			        class_i, get_num_classes(), label_j, get_num_labels(),
			        l_pos);

			// next free element in m_labels[label_j]:
			int32_t c_pos = next_class_idx[label_j];
			REQUIRE(c_pos < m_labels[label_j].size(),
			        "c_pos exceeded length of labels vector");
			next_class_idx[label_j]++;

			// finally, story label_j into class-column
			m_labels[label_j][c_pos] = class_i;
		}
	}
	SG_FREE(next_class_idx);

	return;
}


void
CMultilabelLabels::display() const
{
	SGVector <int32_t> ** labels_list = get_class_labels();
	SG_PRINT("printing %d binary label vectors for %d multilabels:\n",
	         get_num_classes(), get_num_labels());

	for (int32_t class_i = 0; class_i < get_num_classes(); class_i++)
	{
		SG_PRINT("  yC_{class_i=%d}", class_i);
		SGVector <float64_t> dense =
		        to_dense <int32_t, float64_t> (labels_list[class_i],
		                                       get_num_labels(), +1, -1);
		dense.display_vector("");
		delete labels_list[class_i];
	}
	SG_FREE(labels_list);

	SG_PRINT("printing %d binary class vectors for %d labels:\n",
	         get_num_labels(), get_num_classes());

	for (int32_t j = 0; j < get_num_labels(); j++)
	{
		SG_PRINT("  y_{j=%d}", j);
		SGVector <float64_t> dense =
		        to_dense <int16_t , float64_t> (&m_labels[j], get_num_classes(),
		                                        +1, -1);
		dense.display_vector("");
	}
	return;
}
