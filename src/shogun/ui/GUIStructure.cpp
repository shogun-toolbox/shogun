/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <ui/SGInterface.h>
#include <ui/GUIStructure.h>

#include <lib/config.h>
#include <io/SGIO.h>
#include <structure/Plif.h>

using namespace shogun;

CGUIStructure::CGUIStructure(CSGInterface* ui_)
: ui(ui_), m_dp(NULL), m_feature_matrix(NULL),
  m_feature_matrix_sparse1(NULL), m_feature_matrix_sparse2(NULL),
  m_feature_dims(NULL), m_num_positions(0), m_all_positions(0),
  m_content_svm_weights(0), m_num_svm_weights(0),
  m_orf_info(NULL), m_use_orf(true), m_mod_words(NULL)
{
  m_plif_matrix=new CPlifMatrix();
  SG_REF(m_plif_matrix);
}

CGUIStructure::~CGUIStructure()
{
	SG_UNREF(m_plif_matrix);
}
