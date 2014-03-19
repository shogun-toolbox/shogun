#include <shogun/lib/config.h>

// Eigen3 is required for working with this example
#ifdef HAVE_EIGEN3
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <cstdio>
#include <shogun/features/DataGenerator.h>

using namespace shogun;
using namespace Eigen;

float64_t minus_func(float64_t left, float64_t right){
  return left - right;
}
float64_t plus_func(float64_t left, float64_t right){
  return left + right;
}
float64_t times_func(float64_t left, float64_t right){
  return left * right;
}
float64_t divide_func(float64_t left, float64_t right){
  return left / right;
}
float64_t func_wrt_var(float64_t left, float64_t right){
  return left / CMath::sqrt(right);
}

float64_t minus_wrt_square(float64_t left, float64_t right){
  return right * right - left;
}

float64_t add_wrt_square(float64_t left, float64_t right){
  return left * left + right * 2.0;
}

float64_t add_wrt_square_2(float64_t left, float64_t right){
  return left * left + right;
}

float64_t times_wrt_square(float64_t left, float64_t right){
  return left * left * right;
}

float64_t times_wrt_var(float64_t left, float64_t right){
  return left * 0.5 / right;
}

float64_t add_wrt_cubic(float64_t left, float64_t right){
  return left * 2.0 + CMath::pow(right,3.0);
}
float64_t add_wrt_cubic_2(float64_t left, float64_t right){
  return left + CMath::pow(right,3.0);
}

void get_standard_normal_PDF(float64_t &x){
  if(x == CMath::INFTY || x == -CMath::INFTY){
    x = 0.0;
  } else {
    x = -0.5*(x * x + CMath::log(2.0*CMath::PI));
    x = CMath::exp(x);
  }
}
void get_standard_normal_PDF(SGVector<float64_t> &x){
  Map<VectorXd> eigen_x(x.vector, x.vlen);
  ArrayXd tmp = eigen_x.array();
  for(index_t idx = 0; idx < tmp.size(); ++idx){
    get_standard_normal_PDF(tmp[idx]);
  }
  eigen_x = tmp;
}

float64_t func_wrt_normal_pdf(float64_t left, float64_t right){
  get_standard_normal_PDF(left);
  return left / CMath::sqrt(right);
}

float64_t func_wrt_normal_cdf(float64_t self){
  return 0.5 * CStatistics::error_function(self / CMath::sqrt(2.0));
}
void matrix_operation(SGMatrix<float64_t> &result,
                      SGMatrix<float64_t> &base,
                      float64_t (*transform_func)(float64_t)
                     ) {
  assert(result.num_rows == base.num_rows);
  assert(result.num_cols == base.num_cols);
  Map<MatrixXd> eigen_reslut(result.matrix, result.num_rows, result.num_cols); 
  for (index_t r = 0; r < eigen_reslut.rows(); r++) {
    for (index_t c = 0; c < eigen_reslut.cols(); c++) {
      eigen_reslut(r,c) = transform_func(base(r,c));
    }
  }
}

void column_vec_op_row_vec(SGMatrix<float64_t> &result,
                           SGVector<float64_t> &col_vec,
                           SGVector<float64_t> &row_vec,
                           float64_t (*op_func)(float64_t, float64_t)
                          ) {
  assert(result.num_rows == col_vec.size());
  assert(result.num_cols == row_vec.size());
  Map<MatrixXd> eigen_reslut(result.matrix, result.num_rows, result.num_cols); 
  Map<VectorXd> eigen_col_vec(col_vec.vector, col_vec.vlen);
  Map<VectorXd> eigen_row_vec(row_vec.vector, row_vec.vlen);
  for (index_t r = 0; r < eigen_reslut.rows(); r++) {
    for (index_t c = 0; c < eigen_reslut.cols(); c++) {
      eigen_reslut(r,c) = op_func(eigen_col_vec(r), eigen_row_vec(c));
    }
  }
}
void matrix_op_vec(SGMatrix<float64_t> &result,
                   SGMatrix<float64_t> &base,
                   SGVector<float64_t> &vec,
                   float64_t (*op_func)(float64_t, float64_t),
                   bool is_row_vec = true
                  ) {
  if (is_row_vec)
    assert(result.num_cols == vec.size());
  else
    assert(result.num_rows == vec.size());
  assert(result.num_rows == base.num_rows);
  assert(result.num_cols == base.num_cols);

  Map<MatrixXd> eigen_reslut(result.matrix, result.num_rows, result.num_cols); 
  Map<VectorXd> eigen_vec(vec.vector, vec.vlen);
  for (index_t r = 0; r < eigen_reslut.rows(); r++) {
    for (index_t c = 0; c < eigen_reslut.cols(); c++) {
      if (is_row_vec)
        eigen_reslut(r,c) = op_func(base(r,c), eigen_vec(c));
      else
        eigen_reslut(r,c) = op_func(base(r,c), eigen_vec(r));
    }
  }
}
void matrix_op_vec(SGMatrix<float64_t> &result,
                   SGVector<float64_t> &vec,
                   float64_t (*op_func)(float64_t, float64_t),
                   bool is_row_vec = true
                  ) {
  matrix_op_vec(result, result, vec, op_func, is_row_vec);
}



void ellp(SGVector<float64_t> &m,
          SGVector<float64_t> &v,
          SGMatrix<float64_t> &bound,
          SGVector<float64_t> &f,
          SGVector<float64_t> &gm,
          SGVector<float64_t> &gv,
          char mode) {
  for(index_t i = 0; i < v.size(); i++)
    // Normal variance must be strictly positive
    assert(v[i] > 0.0);

  Map<MatrixXd> eigen_bound(bound.matrix, bound.num_rows, bound.num_cols); 
  SGVector<float64_t> c = bound.get_row_vector(0); 
  SGVector<float64_t> b = bound.get_row_vector(1); 
  SGVector<float64_t> a = bound.get_row_vector(2); 
  SGVector<float64_t> l = bound.get_row_vector(3); 
  SGVector<float64_t> h = bound.get_row_vector(4); 

  Map<VectorXd> eigen_l(l.vector, l.vlen);
  Map<VectorXd> eigen_h(h.vector, h.vlen);

  Map<VectorXd> eigen_v(v.vector, v.vlen);
  Map<VectorXd> eigen_m(m.vector, m.vlen);

  // check the dimension
  assert(eigen_l.size()==eigen_h.size());
  assert(eigen_m.size()==eigen_v.size());

  SGMatrix<float64_t> zl(eigen_l.size(), eigen_m.size());
  SGMatrix<float64_t> zh(eigen_l.size(), eigen_v.size());

  // bsxfun(@minus,l,m)
  column_vec_op_row_vec(zl, l, m, minus_func);
  // zl = bsxfun(@times, bsxfun(@minus,l,m), 1./sqrt(v));
  matrix_op_vec(zl, v, func_wrt_var, true);
  // bsxfun(@minus,h,m)
  column_vec_op_row_vec(zh, h, m, minus_func);
  // zh = bsxfun(@times, bsxfun(@minus,h,m), 1./sqrt(v));
  matrix_op_vec(zh, v, func_wrt_var, true);

  SGMatrix<float64_t> pl(eigen_l.size(), eigen_m.size());
  SGMatrix<float64_t> ph(eigen_l.size(), eigen_v.size());

  // pl = bsxfun(@times, normpdf(zl), 1./sqrt(v)); %normal pdf
  matrix_op_vec(pl, zl, v, func_wrt_normal_pdf, true);
  // ph = bsxfun(@times, normpdf(zh), 1./sqrt(v)); %normal pdf
  matrix_op_vec(ph, zh, v, func_wrt_normal_pdf, true);


  SGMatrix<float64_t> cl(eigen_l.size(), eigen_m.size());
  SGMatrix<float64_t> ch(eigen_l.size(), eigen_v.size());

  // cl = 0.5*erf(zl/sqrt(2)); %normal cdf -const
  matrix_operation(cl, zl, func_wrt_normal_cdf);
  // ch = 0.5*erf(zh/sqrt(2)); %normal cdf -cosnt
  matrix_operation(ch, zh, func_wrt_normal_cdf);

  // l(1) = 0; 
  l[0] = 0.0;
  // h(end) = 0; 
  h[h.vlen-1] = 0.0;
  Map<MatrixXd> eigen_cl(cl.matrix, cl.num_rows, cl.num_cols);
  Map<MatrixXd> eigen_ch(ch.matrix, ch.num_rows, ch.num_cols);

  Map<MatrixXd> eigen_pl(pl.matrix, pl.num_rows, pl.num_cols);
  Map<MatrixXd> eigen_ph(ph.matrix, ph.num_rows, ph.num_cols);

  if(mode & 0x1){

    SGMatrix<float64_t> ex0(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_ex0(ex0.matrix, ex0.num_rows, ex0.num_cols);
    eigen_ex0 = eigen_ch - eigen_cl;

    SGMatrix<float64_t> ex1(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_ex1(ex1.matrix, ex1.num_rows, ex1.num_cols);
    eigen_ex1 = eigen_pl - eigen_ph;

    SGMatrix<float64_t> tmp1(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp1(tmp1.matrix, tmp1.num_rows, tmp1.num_cols);
    //eigen_tmp = eigen_ph - eigen_pl;

    matrix_op_vec(ex1, v, times_func, true);
    matrix_op_vec(tmp1, ex0, m, times_func, true);
    eigen_ex1 += eigen_tmp1;

    SGMatrix<float64_t> ex2(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_ex2(ex2.matrix, ex2.num_rows, ex2.num_cols);

    // bsxfun(@plus, l, m)
    column_vec_op_row_vec(tmp1, l, m, plus_func);
    // bsxfun(@plus, h, m)
    column_vec_op_row_vec(ex2, h, m, plus_func);
    // (bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph
    eigen_tmp1 = eigen_tmp1.cwiseProduct(eigen_pl) - eigen_ex2.cwiseProduct(eigen_ph);
    // bsxfun(@times, v, (bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph)
    matrix_op_vec(tmp1, v, times_func, true);

    SGVector<float64_t> tmp2(v.vlen);
    Map<VectorXd> eigen_tmp2(tmp2.vector, tmp2.vlen);
    // (v+m.^2)
    eigen_tmp2 = eigen_m.cwiseProduct(eigen_m) + eigen_v;
    // bsxfun(@times, (v+m.^2), (ch-cl))
    matrix_op_vec(ex2, ex0, tmp2, times_func, true);
    //ex2 = bsxfun(@times, v, (bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph) + bsxfun(@times, (v+m.^2), (ch-cl));
    eigen_ex2 += eigen_tmp1;

    matrix_op_vec(ex2, a, times_func, false);
    matrix_op_vec(ex1, b, times_func, false);
    matrix_op_vec(ex0, c, times_func, false);

    eigen_tmp1 = eigen_ex0 + eigen_ex1 + eigen_ex2;

    Map<VectorXd> eigen_f(f.vector, f.vlen);
    //f = sum(fr,1)';
    eigen_f = eigen_tmp1.colwise().sum();// f
  }
  if(mode & 0x2){

    SGMatrix<float64_t> tmp1(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp1(tmp1.matrix, tmp1.num_rows, tmp1.num_cols);

    SGMatrix<float64_t> tmp2(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp2(tmp2.matrix, tmp2.num_rows, tmp2.num_cols);

    // bsxfun(@plus, l.^2, 2*v)
    column_vec_op_row_vec(tmp1, l, v, add_wrt_square);
    // bsxfun(@plus, h.^2, 2*v)
    column_vec_op_row_vec(tmp2, h, v, add_wrt_square);
    // bsxfun(@plus, l.^2, 2*v).*pl - bsxfun(@plus, h.^2, 2*v).*ph
    eigen_tmp1 = eigen_tmp1.cwiseProduct(eigen_pl) - eigen_tmp2.cwiseProduct(eigen_ph);
    // bsxfun(@times, a, bsxfun(@plus, l.^2, 2*v).*pl - bsxfun(@plus, h.^2, 2*v).*ph)
    matrix_op_vec(tmp1, a, times_func, false);
    //bsxfun(@times, a, m)
    column_vec_op_row_vec(tmp2, a, m, times_func);
    // 2*bsxfun(@times, a, m).*(ch - cl)
    eigen_tmp1 += eigen_tmp2.cwiseProduct(eigen_ch - eigen_cl)*2.0; //gm

    SGMatrix<float64_t> tmp3(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp3(tmp3.matrix, tmp3.num_rows, tmp3.num_cols);
    // bsxfun(@times, l, pl)
    matrix_op_vec(tmp2, pl, l, times_func, false);
    // bsxfun(@times, h, ph)
    matrix_op_vec(tmp3, ph, h, times_func, false);
    // bsxfun(@times, l, pl) - bsxfun(@times, h, ph)
    eigen_tmp2 -= eigen_tmp3;
    // bsxfun(@times, b, bsxfun(@times, l, pl) - bsxfun(@times, h, ph))
    matrix_op_vec(tmp2, b, times_func, false);


    SGMatrix<float64_t> tmp4(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp4(tmp4.matrix, tmp4.num_rows, tmp4.num_cols);
    eigen_tmp4 = eigen_ch - eigen_cl;
    // bsxfun(@times, b, ch-cl)
    matrix_op_vec(tmp4, b, times_func, false);

    eigen_tmp3 = eigen_pl - eigen_ph;
    // bsxfun(@times, c, pl-ph)
    matrix_op_vec(tmp3, c, times_func, false);

    eigen_tmp1 += eigen_tmp2 + eigen_tmp4 + eigen_tmp3;


    Map<VectorXd> eigen_gm(gm.vector, gm.vlen);
    /*gm = sum(gm,1)';*/
    eigen_gm = eigen_tmp1.colwise().sum();// gm

  }
  if(mode & 0x4){
    SGMatrix<float64_t> tmp1(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp1(tmp1.matrix, tmp1.num_rows, tmp1.num_cols);

    SGMatrix<float64_t> tmp2(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp2(tmp2.matrix, tmp2.num_rows, tmp2.num_cols);

    // bsxfun(@times, v, l)
    column_vec_op_row_vec(tmp1, l, v, times_func);
    // bsxfun(@times, l.^2, m)
    column_vec_op_row_vec(tmp2, l, m, times_wrt_square);
    // bsxfun(@plus, 2*bsxfun(@times, v, l), l.^3)
    matrix_op_vec(tmp1, l, add_wrt_cubic, false);
    // t1 = bsxfun(@plus, 2*bsxfun(@times, v, l), l.^3) - bsxfun(@times, l.^2, m)

    eigen_tmp1 -= eigen_tmp2; // t1

    SGMatrix<float64_t> tmp3(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp3(tmp3.matrix, tmp3.num_rows, tmp3.num_cols);

    // bsxfun(@times, v, h)
    column_vec_op_row_vec(tmp2, h, v, times_func);
    // bsxfun(@times, h.^2, m)
    column_vec_op_row_vec(tmp3, h, m, times_wrt_square);
    // bsxfun(@plus, 2*bsxfun(@times, v, h), h.^3)
    matrix_op_vec(tmp2, h, add_wrt_cubic, false);
    // t2 = bsxfun(@plus, 2*bsxfun(@times, v, h), h.^3) - bsxfun(@times, h.^2, m)
    eigen_tmp2 -= eigen_tmp3; // t2

    // bsxfun(@times, a/2, 1./v)
    column_vec_op_row_vec(tmp3, a, v, times_wrt_var);
    // bsxfun(@times, a/2, 1./v).*(t1.*pl - t2.*ph)
    eigen_tmp3 = eigen_tmp3.cwiseProduct(eigen_tmp1.cwiseProduct(eigen_pl) - eigen_tmp2.cwiseProduct(eigen_ph));

    SGMatrix<float64_t> tmp4(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp4(tmp4.matrix, tmp4.num_rows, tmp4.num_cols);
    eigen_tmp4 = eigen_ch - eigen_cl;
    // bsxfun(@times, a, ch-cl)
    matrix_op_vec(tmp4, a, times_func, false);
    eigen_tmp3 += eigen_tmp4;  // gv


    //bsxfun(@plus, l.^2, v)
    column_vec_op_row_vec(tmp4, l, v, add_wrt_square_2);
    //bsxfun(@times, l, m)
    column_vec_op_row_vec(tmp1, l, m, times_func);
    //(bsxfun(@plus, l.^2, v) - bsxfun(@times, l, m)).*pl
    eigen_tmp1 = (eigen_tmp4 - eigen_tmp1).cwiseProduct(eigen_pl);
    //bsxfun(@plus, h.^2, v)
    column_vec_op_row_vec(tmp4, h, v, add_wrt_square_2);
    //bsxfun(@times, h, m)
    column_vec_op_row_vec(tmp2, h, m, times_func);
    //((bsxfun(@plus, l.^2, v) - bsxfun(@times, l, m)).*pl - (bsxfun(@plus, h.^2, v) - bsxfun(@times, h, m)).*ph)
    eigen_tmp1 -= (eigen_tmp4 - eigen_tmp2).cwiseProduct(eigen_ph);


    //  bsxfun(@times, b/2, 1./v)
    column_vec_op_row_vec(tmp2, b, v, times_wrt_var);
    //gv = gv + bsxfun(@times, b/2, 1./v).*...
    //((bsxfun(@plus, l.^2, v) - bsxfun(@times, l, m)).*pl ... 
    //- (bsxfun(@plus, h.^2, v) - bsxfun(@times, h, m)).*ph);
    eigen_tmp3 += eigen_tmp2.cwiseProduct(eigen_tmp1);


    // bsxfun(@times, c/2, 1./v)
    column_vec_op_row_vec(tmp2, c, v, times_wrt_var);
    // bsxfun(@minus,l,m)
    column_vec_op_row_vec(tmp1, l, m, minus_func);
    // bsxfun(@minus,h,m)
    column_vec_op_row_vec(tmp4, h, m, minus_func);
    // (bsxfun(@minus,l,m)).*pl - (bsxfun(@minus,h,m)).*ph
    //bsxfun(@times, c/2, 1./v).*((bsxfun(@minus,l,m)).*pl - (bsxfun(@minus,h,m)).*ph);
    eigen_tmp2 = eigen_tmp2.cwiseProduct(eigen_tmp1.cwiseProduct(eigen_pl) - eigen_tmp4.cwiseProduct(eigen_ph));
    eigen_tmp3 += eigen_tmp2;
    //gv = gv + bsxfun(@times, c/2, 1./v).*...
    /*((bsxfun(@minus,l,m)).*pl - (bsxfun(@minus,h,m)).*ph);*/

    Map<VectorXd> eigen_gv(gv.vector, gv.vlen);
    /*gv = sum(gv,1)';*/
    eigen_gv = eigen_tmp3.colwise().sum();// gv

  }
  if(mode & 0x8){
    SGMatrix<float64_t> tmp1(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp1(tmp1.matrix, tmp1.num_rows, tmp1.num_cols);
    //-l.^2*m + 2*l*v
    eigen_tmp1 = eigen_l * eigen_v.transpose() * 2.0 - (eigen_l.cwiseProduct(eigen_l) * eigen_m.transpose());
    //bsxfun(@plus, l.^3, -l.^2*m + 2*l*v)
    matrix_op_vec(tmp1, l, add_wrt_cubic_2, false);
    //bsxfun(@plus, l.^3, -l.^2*m + 2*l*v).*pl
    eigen_tmp1 = eigen_tmp1.cwiseProduct(eigen_pl);

    SGMatrix<float64_t> tmp2(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp2(tmp2.matrix, tmp2.num_rows, tmp2.num_cols);
    //-h.^2*m + 2*h*v
    eigen_tmp2 = eigen_h * eigen_v.transpose() * 2.0 - (eigen_h.cwiseProduct(eigen_h) * eigen_m.transpose());
    //bsxfun(@plus, h.^3, -h.^2*m + 2*h*v)
    matrix_op_vec(tmp2, h, add_wrt_cubic_2, false);
    //bsxfun(@plus, l.^3, -l.^2*m + 2*l*v).*pl - bsxfun(@plus, h.^3, -h.^2*m + 2*h*v).*ph
    eigen_tmp2 = eigen_tmp1 - eigen_tmp2.cwiseProduct(eigen_ph);
    //bsxfun(@times, 1./v, bsxfun(@plus, l.^3, -l.^2*m + 2*l*v).*pl - bsxfun(@plus, h.^3, -h.^2*m + 2*h*v).*ph)
    matrix_op_vec(tmp2, v, divide_func, true);

    eigen_tmp2 += (eigen_ch - eigen_cl)*2.0;

    //hm = bsxfun(@times, a, bsxfun(@times, 1./v, bsxfun(@plus, l.^3, -l.^2*m + 2*l*v).*pl - bsxfun(@plus, h.^3, -h.^2*m + 2*h*v).*ph) ...
    //+ 2.*(ch - cl));
    matrix_op_vec(tmp2, a, times_func, false); // hm

    SGMatrix<float64_t> tmp3(eigen_l.size(), eigen_m.size());
    Map<MatrixXd> eigen_tmp3(tmp3.matrix, tmp3.num_rows, tmp3.num_cols);
    eigen_tmp3 = eigen_l * eigen_m.transpose();
    //bsxfun(@minus, l.^2, l*m)
    matrix_op_vec(tmp3, l, minus_wrt_square, false);


    eigen_tmp1 = eigen_h * eigen_m.transpose();
    //bsxfun(@minus, h.^2, h*m)
    matrix_op_vec(tmp1, h, minus_wrt_square, false);

    //bsxfun(@minus, l.^2, l*m).*pl - bsxfun(@minus, h.^2, h*m).*ph
    eigen_tmp3 = eigen_tmp3.cwiseProduct(eigen_pl) - eigen_tmp1.cwiseProduct(eigen_ph);
    //bsxfun(@times, 1./v, bsxfun(@minus, l.^2, l*m).*pl - bsxfun(@minus, h.^2, h*m).*ph)
    matrix_op_vec(tmp3, v, divide_func, true);
    eigen_tmp3 += eigen_pl - eigen_ph;

    matrix_op_vec(tmp3, b, times_func, false);
    //hm = hm + bsxfun(@times, b, bsxfun(@times, 1./v, bsxfun(@minus, l.^2, l*m).*pl - bsxfun(@minus, h.^2, h*m).*ph) ...
    //+ (pl - ph));
    eigen_tmp2 += eigen_tmp3;

    column_vec_op_row_vec(tmp1, l, m, minus_func);
    column_vec_op_row_vec(tmp3, h, m, minus_func);
    eigen_tmp1 = eigen_tmp1.cwiseProduct(eigen_pl) - eigen_tmp3.cwiseProduct(eigen_ph);
    column_vec_op_row_vec(tmp3, c, v, divide_func);

    //hm = hm + bsxfun(@times, c, 1./v).*...
    //((bsxfun(@minus,l,m)).*pl - (bsxfun(@minus,h,m)).*ph);
    eigen_tmp2 += eigen_tmp3.cwiseProduct(eigen_tmp1);

    //Map<VectorXd> eigen_gh(gh.vector, gh.vlen);
    //eigen_gh = eigen_tmp2.colwise().sum();// gh
  }
}

void computeElogLik(SGVector<float64_t> &y,
                    SGVector<float64_t> &m,
                    SGVector<float64_t> &v,
                    SGMatrix<float64_t> &bound,
                    SGVector<float64_t> &f,
                    SGVector<float64_t> &gm,
                    SGVector<float64_t> &gv
                   ) {
  Map<VectorXd> eigen_y(y.vector, y.vlen);
  Map<VectorXd> eigen_m(m.vector, m.vlen);

  ellp(m, v, bound, f, gm, gv, 15);

  Map<VectorXd> eigen_f(f.vector, f.vlen);
  Map<VectorXd> eigen_gm(gm.vector, gm.vlen);
  Map<VectorXd> eigen_gv(gv.vector, gv.vlen);
  // f = y.*m - t;
  eigen_f = eigen_y.cwiseProduct(eigen_m) - eigen_f;
  // gm = y - gm;
  eigen_gm = eigen_y - eigen_gm;
  // gv = -gv;
  eigen_gv = -eigen_gv;
}
int main(int argc, char *argv[])
{
  index_t row = 5;
  index_t col = 20;
  SGMatrix<float64_t> bound(row, col);
  bound(0,0)=0.000188712193000;
  bound(0,1)=0.028090310300000;
  bound(0,2)=0.110211757000000;
  bound(0,3)=0.232736440000000;
  bound(0,4)=0.372524706000000;
  bound(0,5)=0.504567936000000;
  bound(0,6)=0.606280283000000;
  bound(0,7)=0.666125432000000;
  bound(0,8)=0.689334264000000;
  bound(0,9)=0.693147181000000;
  bound(0,10)=0.693147181000000;
  bound(0,11)=0.689334264000000;
  bound(0,12)=0.666125432000000;
  bound(0,13)=0.606280283000000;
  bound(0,14)=0.504567936000000;
  bound(0,15)=0.372524706000000;
  bound(0,16)=0.232736440000000;
  bound(0,17)=0.110211757000000;
  bound(0,18)=0.028090310400000;
  bound(0,19)=0.000188712000000;

  bound(1,0)=0;
  bound(1,1)=0.006648614600000;
  bound(1,2)=0.034432684600000;
  bound(1,3)=0.088701969900000;
  bound(1,4)=0.168024214000000;
  bound(1,5)=0.264032863000000;
  bound(1,6)=0.360755794000000;
  bound(1,7)=0.439094482000000;
  bound(1,8)=0.485091758000000;
  bound(1,9)=0.499419205000000;
  bound(1,10)=0.500580795000000;
  bound(1,11)=0.514908242000000;
  bound(1,12)=0.560905518000000;
  bound(1,13)=0.639244206000000;
  bound(1,14)=0.735967137000000;
  bound(1,15)=0.831975786000000;
  bound(1,16)=0.911298030000000;
  bound(1,17)=0.965567315000000;
  bound(1,18)=0.993351385000000;
  bound(1,19)=1.000000000000000;

  bound(2,0)=0;
  bound(2,1)=0.000397791059000;
  bound(2,2)=0.002753100850000;
  bound(2,3)=0.008770186980000;
  bound(2,4)=0.020034759300000;
  bound(2,5)=0.037511596000000;
  bound(2,6)=0.060543032900000;
  bound(2,7)=0.086256780600000;
  bound(2,8)=0.109213531000000;
  bound(2,9)=0.123026104000000;
  bound(2,10)=0.123026104000000;
  bound(2,11)=0.109213531000000;
  bound(2,12)=0.086256780600000;
  bound(2,13)=0.060543032900000;
  bound(2,14)=0.037511596000000;
  bound(2,15)=0.020034759300000;
  bound(2,16)=0.008770186980000;
  bound(2,17)=0.002753100850000;
  bound(2,18)=0.000397791059000;
  bound(2,19)=0;

  bound(3,0)=-CMath::INFTY;
  bound(3,1)=-8.575194939999999;
  bound(3,2)=-5.933689180000000;
  bound(3,3)=-4.525933600000000;
  bound(3,4)=-3.528107790000000;
  bound(3,5)=-2.751548540000000;
  bound(3,6)=-2.097898790000000;
  bound(3,7)=-1.519690830000000;
  bound(3,8)=-0.989533382000000;
  bound(3,9)=-0.491473077000000;
  bound(3,10)=0;
  bound(3,11)=0.491473077000000;
  bound(3,12)=0.989533382000000;
  bound(3,13)=1.519690830000000;
  bound(3,14)=2.097898790000000;
  bound(3,15)=2.751548540000000;
  bound(3,16)=3.528107790000000;
  bound(3,17)=4.525933600000000;
  bound(3,18)=5.933689180000000;
  bound(3,19)=8.575194939999999;


  bound(4,0)=-8.575194939999999;
  bound(4,1)=-5.933689180000000;
  bound(4,2)=-4.525933600000000;
  bound(4,3)=-3.528107790000000;
  bound(4,4)=-2.751548540000000;
  bound(4,5)=-2.097898790000000;
  bound(4,6)=-1.519690830000000;
  bound(4,7)=-0.989533382000000;
  bound(4,8)=-0.491473077000000;
  bound(4,9)=0;
  bound(4,10)=0.491473077000000;
  bound(4,11)=0.989533382000000;
  bound(4,12)=1.519690830000000;
  bound(4,13)=2.097898790000000;
  bound(4,14)=2.751548540000000;
  bound(4,15)=3.528107790000000;
  bound(4,16)=4.525933600000000;
  bound(4,17)=5.933689180000000;
  bound(4,18)=8.575194939999999;
  bound(4,19)= CMath::INFTY;


  const index_t dim = 2;
  SGVector<float64_t> y(dim);
  SGVector<float64_t> m(dim);
  SGVector<float64_t> v(dim);
  SGVector<float64_t> f(dim);
  SGVector<float64_t> gm(dim);
  SGVector<float64_t> gv(dim);
  y[0] = 1;
  y[1] = 1;
  m[0] = 0.5;
  m[1] = 10;
  v[0] = 1;
  v[1] = 1000;

  computeElogLik(y, m, v, bound, f, gm, gv);

  for(index_t ii = 0 ; ii < dim; ++ii ){
    printf("f[%d]=%.10f gm[%d]=%.10f gv[%d]=%.10f\n",ii,f[ii],ii,gm[ii],ii,gv[ii]);
  }
  /*
   * y = y(1:2);
   y(1) = 1;
   y(2) = 1;
   m0 = m0(1:2);
   m0(1) = 0.5;
   m0(2) = 10;
   v = v(1:2);
   v(1) =1;
   v(2) =1000;
   [fi, gmi, gvi] = ElogLik('bernLogit', y, m0, v, bound);
   * result from the Matlab code 
   * fi = 
  -0.581802591979227
  -8.261125462996134
     gmi =
   0.397962641541327
   0.376111328425179
    gvi =
  -0.099501122153405
  -0.005991349522653
   * 
   */
  return 0;
}
#else

int main(int argc, char **argv)
{
  return 0;
}

#endif /* HAVE_EIGEN3 */
