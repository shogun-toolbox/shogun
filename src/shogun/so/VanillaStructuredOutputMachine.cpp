/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/so/VanillaStructuredOutputMachine.h>

using namespace shogun;

CVanillaStructuredOutputMachine::CVanillaStructuredOutputMachine()
: CLinearStructuredOutputMachine()
{
}

CVanillaStructuredOutputMachine::CVanillaStructuredOutputMachine(
		CStructuredModel* model,
		CStructuredLoss*  loss,
		CStructuredLabels* labs,
		CFeatures*         features)
: CLinearStructuredOutputMachine(model, loss, labs, features)
{
}

CVanillaStructuredOutputMachine::~CVanillaStructuredOutputMachine()
{
}

//TODO
bool CVanillaStructuredOutputMachine::train_machine(CFeatures* data)
{
}
