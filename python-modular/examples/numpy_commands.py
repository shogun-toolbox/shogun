import shogun.Features as f
import numpy as N

feat1 = f.ShortFeatures(N.zeros((10,5),N.short))
feat2 = f.WordFeatures(N.zeros((10,5),N.uint16))
feat3 = f.RealFeatures(N.zeros((10,5),N.double))

lab = f.Labels(N.array([1,2.,3]))
