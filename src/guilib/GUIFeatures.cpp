#include "guilib/GUIFeatures.h"

CGUIFeatures::CGUIFeatures(CGUI * gui_)
  : gui(gui_), top_features(NULL, NULL),train_features(&top_features)
{
}

CGUIFeatures::~CGUIFeatures()
{
}
