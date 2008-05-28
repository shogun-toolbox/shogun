from shogun.Features import SparseRealFeatures
f=SparseRealFeatures()
lab=f.load_svmlight_file('testread.light')
f.write_svmlight_file('testwrite.light', lab)
