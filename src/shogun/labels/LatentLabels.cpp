/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/features/LatentLabels.h>

using namespace shogun;

CLatentData::CLatentData ()
{

}

CLatentData::~CLatentData ()
{

}

CLatentLabels::CLatentLabels ()
  : CLabels ()
{
  init ();
}

CLatentLabels::CLatentLabels (int32_t num_labels)
  : CLabels (num_labels)
{
  init ();
  m_labels = new CDynamicObjectArray<CLatentData> (num_labels);
  SG_REF (m_labels);
}

CLatentLabels::~CLatentLabels ()
{
  SG_UNREF (m_labels);
}

void CLatentLabels::init ()
{
  SG_ADD((CSGObject**) &m_labels, "m_labels", "The labels", MS_NOT_AVAILABLE);
  m_labels = NULL;
}

CDynamicObjectArray<CLatentData>* CLatentLabels::get_labels () const
{
  SG_REF (m_labels);
  return m_labels;
}

CLatentData* CLatentLabels::get_latent_label (int32_t idx) const
{
  ASSERT (m_labels != NULL);
  if (idx < 0 || idx >= get_num_labels())
    SG_ERROR("Out of index!\n");

  return (CLatentData*) m_labels->get_element (idx);
}

void CLatentLabels::add_latent_label (CLatentData* label)
{
  ASSERT (m_labels != NULL);
  m_labels->push_back (label);
}

bool CLatentLabels::set_latent_label (int32_t idx, CLatentData* label)
{
  if (idx < get_num_labels ())
  {
    return m_labels->set_element (label, idx);
  }
  else
  {
    return false;
  }
}

int32_t CLatentLabels::get_num_labels() const
{
  if (m_labels == NULL)
    return 0;
  else
    return m_labels->get_num_elements ();
}

