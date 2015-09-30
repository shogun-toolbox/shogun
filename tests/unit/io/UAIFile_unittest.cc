#include <shogun/io/UAIFile.h>
#include <shogun/base/init.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(UAIFileTest, preamble)
{
    CUAIFile* fout = new CUAIFile("UAIFileTest_unittest_preamble.uai", 'w', NULL);
    
    fout->set_net_type("MARKOV");
    
    fout->set_num_vars(3);
    
    SGVector<int32_t> vars_card(3);
    vars_card[0] = 2;
    vars_card[1] = 2;
    vars_card[2] = 3;
    fout->set_vars_card(vars_card);

    fout->set_num_factors(2);

    SGVector<int32_t>* factors_scope = new SGVector<int32_t>[2];               
    SGVector<int32_t> f_s1(2);                                                 
    f_s1[0] = 0;                                                               
    f_s1[1] = 1;                                                               
    factors_scope[0] = f_s1;                                                   
    SGVector<int32_t> f_s2(2);                                                 
    f_s2[0] = 1;                                                               
    f_s2[1] = 2;                                                               
    factors_scope[1] = f_s2;                                                   
    fout->set_factors_scope(2, factors_scope);    

    SGVector<float64_t>* factors_table = new SGVector<float64_t> [2];           
    SGVector<float64_t> f_t1(4);                                              
    f_t1[0] = 0.2;                                                             
    f_t1[1] = 2.2;                                                             
    f_t1[2] = 3.2;                                                             
    f_t1[3] = 4.2;                                                                                                                                            
    SGVector<float64_t> f_t2(6);                                              
    f_t2[0] = 0.2;                                                             
    f_t2[1] = 2.2;                                                             
    f_t2[2] = 3.2;                                                             
    f_t2[3] = 4.2;                                                             
    f_t2[4] = 5.2;                                                             
    f_t2[5] = 6.2;                                                                                                                                         
    factors_table[0] = f_t1;                                                    
    factors_table[1] = f_t2;                                                    
    fout->set_factors_table(2, factors_table);  

    SG_UNREF(fout);
    
    CUAIFile* fin = new CUAIFile("UAIFileTest_unittest_preamble.uai", 'r', NULL);

    SGVector<char> net_type;
    int32_t num_factors, num_vars;
    SGVector<int32_t> vars_card_in;
    SGVector<int32_t>* factors_scope_in;

    fin->parse();

    fin->get_preamble(net_type,
                      num_vars,
                      vars_card_in,
                      num_factors,
                      factors_scope_in);

    const char net_type_expected[] = "MARKOV";
    EXPECT_EQ(net_type.vlen, strlen(net_type_expected));
    for (int32_t i=0; i<net_type.vlen; i++)
        EXPECT_EQ(net_type[i], net_type_expected[i]);

    EXPECT_EQ(num_vars, 3);
    EXPECT_EQ(vars_card_in.vlen, 3);
    for (int32_t i=0; i<num_vars; i++)
        EXPECT_EQ(vars_card_in[i], vars_card[i]);
    EXPECT_EQ(num_factors, 2);
    for (int32_t i=0; i<num_factors; i++)
    {
        SGVector<int32_t> scope = factors_scope[i];
        SGVector<int32_t> scope_in = factors_scope_in[i];
        EXPECT_EQ(scope.vlen, scope_in.vlen);
        for (int32_t j=0; j<scope.vlen; j++)
            EXPECT_EQ(scope_in[j], scope[j]);
    }

    SGVector<float64_t>* factors_table_in;

    fin->get_factors_table(factors_table_in);

    for (int32_t i=0; i<2; i++)
    {
        SGVector<float64_t> table = factors_table[i];
        SGVector<float64_t> table_in = factors_table_in[i];
        EXPECT_EQ(table.vlen, table_in.vlen);
        for (int32_t j=0; j<table.vlen; j++)
            EXPECT_NEAR(table_in[j], table[j], 1E-14);
    }

    SG_UNREF(fin);

    delete [] factors_table_in;
    delete [] factors_scope_in;
    delete [] factors_scope;
    delete [] factors_table;

    unlink("UAIFileTest_unittest_preamble.uai");
}

