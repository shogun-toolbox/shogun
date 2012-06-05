/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#ifndef __LATENTLABELS_H__
#define __LATENTLABELS_H__

#include <shogun/features/Labels.h>
#include <shogun/lib/DynamicObjectArray.h>

namespace shogun
{
  class CLatentData : public CSGObject
  {
    public:
      CLatentData ();

      virtual ~CLatentData ();

      virtual const char* get_name() const { return "LatentData"; }
  };

  class CLatentLabels : public CLabels
  {
    public:
      CLatentLabels ();

      CLatentLabels (int32_t num_labels);

      virtual ~CLatentLabels ();

      CDynamicObjectArray<CLatentData>* get_labels() const;

      CLatentData* get_label (int32_t idx) const;

      void add_label (CLatentData* label);

      using CLabels::set_label;

      bool set_label (int32_t idx, CLatentData* label);

      int32_t get_num_labels() const;

      virtual const char* get_name() const { return "LatentLabels"; }

    protected:
      /** the vector of labels */
      CDynamicObjectArray<CLatentData>* m_labels;

    private:
      void init ();
  };
}

#endif /* __LATENTLABELS_H__ */

