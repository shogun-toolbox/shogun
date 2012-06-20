/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/StructuredLabels.h>
#include <shogun/lib/common.h>
#include <shogun/loss/HingeLoss.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/MulticlassRiskFunction.h>

using namespace shogun;

#define	DIMS		5
#define EPSILON  	0
#define	NUM_SAMPLES	100
#define NUM_CLASSES	3

void gen_rand_data(SGVector< float64_t > labs, SGMatrix< float64_t > feats)
{
    float64_t means[DIMS];
    float64_t  stds[DIMS];

    for ( int32_t c = 0 ; c < NUM_CLASSES ; ++c )
    {
        for ( int32_t j = 0 ; j < DIMS ; ++j )
        {
            means[j] = CMath::random(-1, 1);
             stds[j] = CMath::random( 1, 5);
        }

        for ( int32_t i = 0 ; i < NUM_SAMPLES ; ++i )
        {
            labs[c*NUM_SAMPLES+i] = c;

            for ( int32_t j = 0 ; j < DIMS ; ++j )
            {
                feats[(c*NUM_SAMPLES+i)*DIMS + j] =
                    CMath::normal_random(means[j], stds[j]);
            }
        }
    }
}

void print_message(FILE* target, const char* str)
{
    fprintf(target, "%s", str);
}

int main(int argc, char * argv[])
{
    init_shogun_with_defaults();

    SGVector< float64_t > labs(NUM_CLASSES*NUM_SAMPLES);
    SGMatrix< float64_t > feats(DIMS, NUM_CLASSES*NUM_SAMPLES);

    gen_rand_data(labs, feats);

    // Create train labels
    CMulticlassSOLabels* labels = new CMulticlassSOLabels(labs);

    // Create train features
    CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feats);

    // Create structured model
    CMulticlassModel* model = new CMulticlassModel(features, labels);

    // Create loss function
    CHingeLoss* loss = new CHingeLoss();

    // Create risk function
    CMulticlassRiskFunction* risk = new CMulticlassRiskFunction();

    // Create SO-SVM
    CDualLibQPBMSOSVM* sosvm = new CDualLibQPBMSOSVM(model, loss, labels, features, 0.01, risk);
    SG_REF(sosvm);

    sosvm->train();

    SG_SPRINT("\n");
    SGVector< float64_t > w = sosvm->get_w();
    for ( int32_t i = 0 ; i < w.vlen ; ++i )
        SG_SPRINT("%10f ", w[i]);
    SG_SPRINT("\n\n");

    // Free memory
    SG_UNREF(sosvm);

    exit_shogun();

	return 0;
}
