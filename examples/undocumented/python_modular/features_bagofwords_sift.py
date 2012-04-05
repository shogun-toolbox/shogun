from shogun.Features import LongIntFeatures
import numpy
from numpy.random import randn, randint
from numpy import int64, array, hstack, vstack

data=[]			# fabricated data (not read from file)
dictionary=[randn(127).tolist(), randn(127).tolist()]	# contains unique sift vectors from all of the images

def fabricateData():
	'''
	The user might swap this function for either a dense sift extractor,
	or an interest (key) point sift extractor
	'''
	for i in range(0,5):
		row=[]
		for j in range(0,randint(2,5)):
			sift=randn(127)
			row.extend(sift.tolist())
		global data
		data.append(row)
				
def createDictionary():
	'''
	For all the sift vectors in the database, we need to create the dictionary a-k-a the bag of words.
	This dicitonary should not contain duplicate entries.
	'''
	#print len(data)	
	for image in data:
		# assuming the sift vectors were extracted at interest points 
		# as opposed to a dense extraction
		# i.e. the number of sift descriptors in each image need not be the same

		InterestPoints=len(image)/127
		siftVectors=[]

		for i in range(0,InterestPoints):
			siftVectors.append(image[i*127:(i+1)*127-1])

		StoreUniqueVectors(siftVectors)
				

def StoreUniqueVectors(siftVectors):
	#print 'Booyay!'	 
	
	for vector in siftVectors:
		isUnique=True
		
		for i in dictionary:
			x=(i==vector)
			if(x):
				isUnique=False
		
		if(isUnique):
			dictionary.append(vector)
	
	

def createShogunFeat():
	#print 'hi'
	feature=[]		
	for image in data:
		InterestPoints=len(image)/127
		siftVectors=[]

		for i in range(0,InterestPoints):
			siftVectors.append(image[i*127:(i+1)*127-1])
		feature.append(createHistogram(siftVectors))
	
	B=array(feature,dtype=int64)
	#print B
	shogunFeat=LongIntFeatures(B)
	#shogunMatrix=shogunFeat.get_feature_matrix(shogunFeat)
	return shogunFeat


def createHistogram(siftVectors):
	#print 'booyay'
	histogram= []

	for i in dictionary:
		count=0
		for vector in siftVectors:
			
			if(i==vector):
				count=count+1
		histogram.append(count)
	print histogram	
	return histogram

def main():	
	fabricateData()
	createDictionary()
	shogunFeat=createShogunFeat()

	
if __name__=='__main__':
	main()

