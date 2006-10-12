import features.Features as f
import numpy as N

feat1 = f.CharFeatures(N.chararray((10,5),1),f.DNA)
feat2 = f.ShortFeatures(N.zeros((10,5),N.short))
feat3 = f.WordFeatures(N.zeros((10,5),N.uint16))
feat4 = f.RealFeatures(N.zeros((10,5),N.double))
feat5 = f.ByteFeatures(N.zeros((10,5),N.uint8),f.DNA)

lab = f.Labels(N.array([1,2.,3]))
#lab = f.CLabels(N.array([1,2.,3]))
