/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/tree/CHAIDTree.h>

using namespace shogun;

const float64_t CCHAIDTree::MISSING=CMath::NOT_A_NUMBER;

CCHAIDTree::CCHAIDTree()
: CTreeMachine<CHAIDTreeNodeData>()
{
	init();
}

CCHAIDTree::~CCHAIDTree()
{
}

void CCHAIDTree::set_machine_problem_type(EProblemType mode)
{
	m_mode=mode;
}

bool CCHAIDTree::is_label_valid(CLabels* lab) const
{
	if (m_mode==PT_MULTICLASS && lab->get_label_type()==LT_MULTICLASS)
		return true;
	else if (m_mode==PT_REGRESSION && lab->get_label_type()==LT_REGRESSION)
		return true;
	else
		return false;
}

CMulticlassLabels* CCHAIDTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")
	return new CMulticlassLabels(); 
}

CRegressionLabels* CCHAIDTree::apply_regression(CFeatures* data)
{
	REQUIRE(data, "Data required for regression in apply_regression\n")
	return new CRegressionLabels();
}

void CCHAIDTree::set_weights(SGVector<float64_t> w)
{
	m_weights=w;
}

SGVector<float64_t> CCHAIDTree::get_weights() const
{
	return m_weights;
}

void CCHAIDTree::clear_weights()
{
	m_weights=SGVector<float64_t>();
}

void CCHAIDTree::set_feature_types(SGVector<int32_t> ft)
{
	m_feature_types=ft;
}

SGVector<int32_t> CCHAIDTree::get_feature_types() const
{
	return m_feature_types;
}

void CCHAIDTree::clear_feature_types()
{
	m_feature_types=SGVector<int32_t>();
}

bool CCHAIDTree::train_machine(CFeatures* data)
{
	REQUIRE(data, "Data required for training\n")
	return true;
}

void CCHAIDTree::init()
{
	m_feature_types=SGVector<int32_t>();
	m_weights=SGVector<float64_t>();
	m_mode=PT_MULTICLASS;

	SG_ADD(&m_weights,"m_weights", "weights", MS_NOT_AVAILABLE);
	SG_ADD(&m_feature_types,"m_feature_types", "feature types", MS_NOT_AVAILABLE);
}
