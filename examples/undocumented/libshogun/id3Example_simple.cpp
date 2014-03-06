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
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/multiclass/tree/ID3ClassifierTree.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	SGMatrix<float64_t> data(2,6);
	data(0,0)=1.0;
	data(1,0)=2.0;
	data(0,1)=1.0;
	data(1,1)=3.0;
	data(0,2)=1.0;
	data(1,2)=4.0;
	data(0,3)=4.0;
	data(1,3)=2.0;
	data(0,4)=4.0;
	data(1,4)=2.0;
	data(0,5)=4.0;
	data(1,5)=3.0;

	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(data);

	SGVector<float64_t> lab(6);
	lab[0] = 11.0;
	lab[1] = 11.0;
	lab[2] = 11.0;
	lab[3] = 22.0;
	lab[4] = 22.0;
	lab[5] = 33.0;

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	CID3ClassifierTree* id3 = new CID3ClassifierTree();
	id3->set_labels(labels);
	id3->train(feats);

	SGMatrix<float64_t> test(2,4);
	test(0,0)=14.0;
	test(1,0)=2.0;
	test(0,1)=4.0;
	test(1,1)=3.0;
	test(0,2)=4.0;
	test(1,2)=2.0;
	test(0,3)=1.0;
	test(1,3)=4.0;

	CDenseFeatures<float64_t>* test_feats = new CDenseFeatures<float64_t>(test);

	CMulticlassLabels* result = (CMulticlassLabels*) id3->apply(test_feats);
	
	SGVector<float64_t>::display_vector(result->get_labels().vector, 
					result->get_labels().vlen, "result");

	exit_shogun();

	return 0;
}
