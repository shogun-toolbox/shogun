#include <set>
#include <map>
#include <vector>
#include <algorithm>

#include <shogun/evaluation/ClusteringEvaluation.h>
#include <shogun/mathematics/munkres.h>

using namespace shogun;
using namespace std;

static void unique_labels(CLabels* labels, vector<int32_t>& result)
{
    set<int32_t> uniq_lbl;
    for (int i = labels->get_num_labels() - 1; i >= 0; --i) {
        uniq_lbl.insert(labels->get_int_label(i));
    }
    result.assign(uniq_lbl.begin(), uniq_lbl.end());
}

static int find_match_count(CLabels* l1, int m1, CLabels* l2, int m2)
{
    int match_count = 0;
    for (int i = l1->get_num_labels() - 1; i >= 0; --i) {
        if (l1->get_int_label(i) == m1 && l2->get_int_label(i) == m2)
            match_count++;
    }
    return match_count;
}

void CClusteringEvaluation::best_map(CLabels* predicted, CLabels* ground_truth)
{
    ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels());
    vector<int32_t> label_p, label_g;
    unique_labels(predicted, label_p);
    unique_labels(ground_truth, label_g);

    int n_class = max(label_p.size(), label_g.size());
    SGMatrix<double> G(n_class, n_class);
    G.zero();

    for (size_t i = 0; i < label_g.size(); ++i) {
        for (size_t j = 0; j < label_p.size(); ++j) {
            G(i, j) = -find_match_count(ground_truth, label_g[i], predicted, label_p[j]) 
                + ground_truth->get_num_labels(); // add a constant since our munkres lib do
                                                  // not support negative cost value
        }
    }

    for (size_t i = 0; i < label_g.size(); ++i) {
        for (size_t j = 0; j < label_p.size(); ++j) {
            printf("%5.0lf ", G(i, j));
        }
        printf("\n");
    }

    Munkres munkres_solver(G);
    munkres_solver.solve();

    for (size_t i = 0; i < label_g.size(); ++i) {
        for (size_t j = 0; j < label_p.size(); ++j) {
            printf("%3.0lf ", G(i, j));
        }
        printf("\n");
    }

    map<int, int> label_map;
    for (size_t i = 0; i < label_p.size(); ++i) {
        for (size_t j = 0; j < label_g.size(); ++j) {
            if (G(j, i) == 0) {
                label_map.insert(make_pair(label_p[i], label_g[j]));
                break;
            }
        }
    }
    for (map<int,int>::iterator it = label_map.begin(); it != label_map.end(); ++it) {
        printf("%d => %d\n", it->first, it->second);
    }

    for (int i = predicted->get_num_labels()-1; i >= 0; --i) {
        predicted->set_int_label(i, label_map[predicted->get_int_label(i)]);
    }

    G.destroy_matrix();
}
