/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Copyright (C) 2014 Jiaolong Xu
 */
#include <shogun/io/SGIO.h>
#include <shogun/io/LineReader.h>
#include <shogun/io/Parser.h>
#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/Time.h>

#include <shogun/mathematics/Math.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>
#include <shogun/structure/StochasticSOSVM.h>
#include <shogun/structure/FactorType.h>
#include <shogun/structure/MAPInference.h>
#include <shogun/structure/FactorGraphModel.h>
#include <shogun/features/FactorGraphFeatures.h>
#include <shogun/labels/FactorGraphLabels.h>

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

using namespace shogun;

#define NUM_STATUS 2
#define NUM_CLASSES 6
#define USE_RANDOM_DATA
// #define SHOW_DATA

const char FNAME_TRAIN[] = "../../../data/multilabel/scene_train.txt";
const char FNAME_TEST[]  = "../../../data/multilabel/scene_test.txt";

void gen_data(SGMatrix< int32_t >& labels, SGMatrix< float64_t >& feats)
{
    int32_t dim		     = 2;
    int32_t num_sample	 = 100;
    int32_t num_classes =	6;
    
    labels = SGMatrix<int32_t>(num_classes, num_sample);
    feats  = SGMatrix<float64_t>(dim, num_sample);
    
    for(int32_t i = 0; i < num_sample; i++)
    {
        // generate labels
        for(int32_t j = 0; j < num_classes; j++)
        {
            labels[i*num_classes + j] = CMath::random(0, 1);
        }
        // generate feature
        for(int32_t j = 0; j < dim; j++)
        {
            feats[i*dim + j] = CMath::random(0.1, 5.0);
        }
    }
}

int32_t read_numbers(SGVector<char> line, std::vector<float64_t> &v, CParser* parser_line) 
{
    parser_line->set_text(line);
    
    int32_t num_entries = 0;
    while (parser_line->has_next())
    {
         v.push_back(parser_line->read_real());
         num_entries++;
    }
    
    return num_entries;
}

void read_data(const char* fname, SGMatrix<int32_t>& labels, SGMatrix<float64_t>& feats)
{
    FILE* pfile  = fopen(fname, "r");
    if (pfile == NULL)
        SG_SERROR("Unable to open file: %s\n", fname);
	
    CDelimiterTokenizer* token_txt = new CDelimiterTokenizer();
    token_txt->delimiters['\n'] = 1;
    SG_REF(token_txt);

    CLineReader* reader_txt = new CLineReader(pfile, token_txt);

    CDelimiterTokenizer* token_line = new CDelimiterTokenizer(true);
    token_line->delimiters[' ']=1;
    SG_REF(token_line);
    
    CParser* parser_line = new CParser();
    parser_line->set_tokenizer(token_line);
    SG_REF(parser_line);
        
    int32_t num_sample = 0;
    SGVector<char> str_line;
    std::vector<float64_t> v_line;
    std::vector<float64_t> v_feats;
    std::vector<int32_t> v_labs;
    while (reader_txt->has_next())
    {
        v_line.clear();
        str_line = reader_txt->read_line();
        int32_t len = read_numbers(str_line, v_line, parser_line);
        for(int32_t i = 0; i < len; i++)
        {
             if(i < NUM_CLASSES)
                 v_labs.push_back((int32_t)v_line[i]);
	     else
		 v_feats.push_back(v_line[i]);
        }    
        num_sample++;
    }
    SG_SPRINT("Loaded %d samples.\n", num_sample);
	
    int32_t dim = v_line.size() - NUM_CLASSES;
    feats  = SGMatrix<float64_t>(dim, num_sample);
    labels = SGMatrix<int32_t>(NUM_CLASSES, num_sample);
    memcpy(&feats[0], &v_feats[0], dim*num_sample*sizeof(float64_t));
    memcpy(&labels[0], &v_labs[0], NUM_CLASSES*num_sample*sizeof(int32_t));
    
    v_feats  = std::vector<float64_t>();
    v_labs   = std::vector<int32_t>();
    str_line = SGVector<char>();
    SG_UNREF(reader_txt);
    SG_UNREF(token_txt);
    SG_UNREF(token_line);
    SG_UNREF(parser_line);
    
    fclose(pfile);
}

SGMatrix< int32_t > get_tree_index()
{
    // A tree structure is defined by a 2-d matrix where
    // each row stores the indecies of a pair of connect factors
    // Define label tree structure
    SGMatrix< int32_t > label_tree_index(5, 2);
    label_tree_index[0] = 0;
    label_tree_index[1] = 0;
    label_tree_index[2] = 1;
    label_tree_index[3] = 4;
    label_tree_index[4] = 2;
	
	label_tree_index[5] = 2;
    label_tree_index[6] = 3;
    label_tree_index[7] = 4;
    label_tree_index[8] = 5;
    label_tree_index[9] = 5;
	
    return label_tree_index;	
}

void build_factor_graph(SGMatrix<float64_t> feats, SGMatrix<int32_t> labels, \
                        CFactorGraphFeatures* fg_feats, CFactorGraphLabels* fg_labels, \
                        std::vector<CTableFactorType*> v_ftp_u, std::vector<CTableFactorType*> v_ftp_t)
{
    int32_t num_sample        = labels.num_cols;
    int32_t num_classes       = labels.num_rows;
    int32_t dim               = feats.num_rows;

    SGMatrix< int32_t > tree_index = get_tree_index();
    int32_t num_edges = tree_index.num_rows;

    // prepare features and labels in factor graph
    for(int32_t n = 0; n < num_sample; n++)
    {
        SGVector<int32_t> vc(num_classes);
        SGVector<int32_t>::fill_vector(vc.vector, vc.vlen, NUM_STATUS);

        CFactorGraph* fg = new CFactorGraph(vc);

        float64_t* pfeat = feats.get_column_vector(n);
        SGVector<float64_t> feat_i(dim);
        memcpy(&feat_i[0], &pfeat[0], dim*sizeof(float64_t));
        // add unary factors
        for(int32_t u=0; u < num_classes; u++)
        {
            SGVector<int32_t> var_index_u(1);
            var_index_u[0] = u;
            CFactor* fac_u = new CFactor(v_ftp_u[u], var_index_u, feat_i);
            fg->add_factor(fac_u);
        }
        // add tree-structred factors
        for(int32_t t = 0; t < num_edges; t++)
        {
            SGVector<float64_t> data_t(1);
            data_t[0] = 1.0;
            SGVector<int32_t> var_index_t = tree_index.get_row_vector(t);
            CFactor* fac_t = new CFactor(v_ftp_t[t], var_index_t, data_t);
            fg->add_factor(fac_t);
        }
        // add factor graph instance
        fg_feats->add_sample(fg);

        // add label
        int32_t* plabs = labels.get_column_vector(n);
        SGVector<int32_t> states_gt(num_classes);
        memcpy(&states_gt[0], &plabs[0], num_classes*sizeof(int32_t));
        SGVector<float64_t> loss_weights(num_classes);
        SGVector<float64_t>::fill_vector(loss_weights.vector, loss_weights.vlen, 1.0);
        CFactorGraphObservation* fg_obs = new CFactorGraphObservation(states_gt, loss_weights);
        fg_labels->add_label(fg_obs);

#ifdef SHOW_DATA
        // show labels
        CFactorGraphObservation* fg_observ = CFactorGraphObservation::obtain_from_generic(fg_labels->get_label(n));
        SG_SPRINT("- sample %d:\n", n);
        SGVector<int32_t> fst = fg_observ->get_data();
        SGVector<int32_t>::display_vector(fst.vector, fst.vlen);
        SGVector<float64_t>::display_vector(feat_i.vector, feat_i.vlen);
        SG_UNREF(fg_observ);
#endif
    }
}

void test()
{
    // Read training data
    SGMatrix<int32_t> labels_train;
    SGMatrix<float64_t> feats_train;
#ifdef USE_RANDOM_DATA
    gen_data(labels_train, feats_train);
#else
    read_data(FNAME_TRAIN, labels_train, feats_train);
#endif
    
    int32_t num_sample_train  = labels_train.num_cols;
    int32_t num_classes       = labels_train.num_rows;
    int32_t dim               = feats_train.num_rows;
    
    // Build factor graph
    SGMatrix< int32_t > tree_index = get_tree_index();
    int32_t num_edges = tree_index.num_rows;
    
    int32_t tid;
    // we have l = num_classes different weights: w_1, w_2, ..., w_l
    // so we create num_classes different unary potentials
    std::vector<CTableFactorType*> v_ftp_u(num_classes);
    for(int32_t u = 0; u < num_classes; u++)
    {
        tid = u;
        SGVector<int32_t> card_u(1);
        card_u[0] = NUM_STATUS;
        SGVector<float64_t> w_u(dim*NUM_STATUS);
        w_u.zero();
        v_ftp_u[u] = new CTableFactorType(tid, card_u, w_u);
        SG_REF(v_ftp_u[u]);
    }
    // define factor type: tree edge factor
    // note that each edge is a new type
    std::vector<CTableFactorType*> v_ftp_t(num_edges);
    for(int32_t t = 0; t < num_edges; t++)
    {
        tid = t + num_classes;
        SGVector<int32_t> card_t(2);
        card_t[0] = NUM_STATUS;
        card_t[1] = NUM_STATUS;
	SGVector<float64_t> w_t(NUM_STATUS*NUM_STATUS);
	w_t.zero();
        v_ftp_t[t] = new CTableFactorType(tid, card_t, w_t);
        SG_REF(v_ftp_t[t]);
    }
	
    // prepare features and labels in factor graph
    CFactorGraphFeatures* fg_feats_train = new CFactorGraphFeatures(num_sample_train);
    SG_REF(fg_feats_train);
    CFactorGraphLabels* fg_labels_train = new CFactorGraphLabels(num_sample_train);
    SG_REF(fg_labels_train);

    build_factor_graph(feats_train, labels_train, fg_feats_train, fg_labels_train, v_ftp_u, v_ftp_t);

    SG_SPRINT("----------------------------------------------------\n");

    CFactorGraphModel* model = new CFactorGraphModel(fg_feats_train, fg_labels_train, TREE_MAX_PROD, false);
    SG_REF(model);

    // initialize model parameters
    for(int32_t u = 0; u < num_classes; u++)
         model->add_factor_type(v_ftp_u[u]);
    for(int32_t t = 0; t < num_edges; t++)
        model->add_factor_type(v_ftp_t[t]);

    // create SGD solver
    CStochasticSOSVM* sgd = new CStochasticSOSVM(model, fg_labels_train);
    sgd->set_num_iter(100);
    sgd->set_lambda(0.01);
    SG_REF(sgd);
	
    // timer
    CTime start;
    // train SGD
    sgd->train();
    float64_t t2 = start.cur_time_diff(false);

    SG_SPRINT(">>>> SGD trained in %9.4f\n", t2);

#ifdef SHOW_DATA
    // check w
    sgd->get_w().display_vector("w_sgd");
#endif

    // Evaluation SGD
    CStructuredLabels* labels_sgd = CLabelsFactory::to_structured(sgd->apply());
    SG_REF(labels_sgd);

    float64_t acc_loss_sgd = 0.0;
    float64_t ave_loss_sgd = 0.0;

    for (int32_t i=0; i<num_sample_train; ++i)
    {
	CStructuredData* y_pred = labels_sgd->get_label(i);
        CStructuredData* y_truth = fg_labels_train->get_label(i);
	acc_loss_sgd += model->delta_loss(y_truth, y_pred);
	SG_UNREF(y_pred);
	SG_UNREF(y_truth);
    }

    ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_sample_train);
    SG_SPRINT("sgd solver: average training loss = %f\n", ave_loss_sgd);

#ifdef USE_RANDOM_DATA
#else
    // Read testing data
    SGMatrix<int32_t> labels_test;
    SGMatrix<float64_t> feats_test;
    read_data(FNAME_TEST, labels_test, feats_test);

    // prepare features and labels in factor graph
    int32_t num_sample_test  = labels_test.num_cols;
    CFactorGraphFeatures* fg_feats_test = new CFactorGraphFeatures(num_sample_test);
    SG_REF(fg_feats_test);
    CFactorGraphLabels* fg_labels_test = new CFactorGraphLabels(num_sample_test);
    SG_REF(fg_labels_test);
    build_factor_graph(feats_test, labels_test, fg_feats_test, fg_labels_test, v_ftp_u, v_ftp_t);

    sgd->set_features(fg_feats_test);
    sgd->set_labels(fg_labels_test);
    labels_sgd = CLabelsFactory::to_structured(sgd->apply());

    acc_loss_sgd = 0.0;
    for (int32_t i=0; i<num_sample_test; ++i)
    {
        CStructuredData* y_pred  = labels_sgd->get_label(i);
        CStructuredData* y_truth = fg_labels_test->get_label(i);
        acc_loss_sgd += model->delta_loss(y_truth, y_pred);
        SG_UNREF(y_pred);
        SG_UNREF(y_truth);
    }

    ave_loss_sgd = acc_loss_sgd / static_cast<float64_t>(num_sample_test);
    SG_SPRINT("sgd solver: average testing error = %f\n", ave_loss_sgd);

    SG_UNREF(fg_feats_test);
    SG_UNREF(fg_labels_test);
#endif

    SG_UNREF(labels_sgd);
    SG_UNREF(sgd);
	SG_UNREF(model);
    SG_UNREF(fg_feats_train);
    SG_UNREF(fg_labels_train);
    for(int32_t u = 0; u < num_classes; u++)
        SG_UNREF(v_ftp_u[u]);
    for(int32_t t = 0; t < num_edges; t++)
        SG_UNREF(v_ftp_t[t]);
}

int main(int argc, char * argv[])
{
    init_shogun_with_defaults();

    //sg_io->set_loglevel(MSG_DEBUG);
	
    test();
    
    exit_shogun();

    return 0;
}
