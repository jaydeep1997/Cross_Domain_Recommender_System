from math import *

# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
	#get the shared items
	si={}		#if s[item]==1, that means item is common in both person1 and person2
	
	for item in prefs[person1]:
		if item in prefs[person2]:
			si[item]=1


	#if no item is common
	if len(si)==0:
		return 0
	
	#sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in prefs[person1] if item in prefs[person2]])
	sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2) for item in si])

	return 1/(1+sum_of_squares)


# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
	# Get the list of mutually rated items
	si={}
	for item in prefs[p1]:
		if item in prefs[p2]:
			si[item]=1

	# Find the number of elements
	n=len(si)
	# if they are no ratings in common, return 0
	if n==0:
		return 0

	# Add up all the preferences
	sum1=sum([prefs[p1][it] for it in si])
	sum2=sum([prefs[p2][it] for it in si])

	# Sum up the squares
	sum1Sq=sum([pow(prefs[p1][it],2) for it in si])
	sum2Sq=sum([pow(prefs[p2][it],2) for it in si])

	# Sum up the products
	pSum=sum([prefs[p1][it]*prefs[p2][it] for it in si])

	# Calculate Pearson score
	num=pSum-(sum1*sum2/n)
	den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
	if den==0:
		return 0
	r=num/den
	return r



# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson,n=100):

	totals={}
	simSums={}

	for other in prefs:
		# don't compare me to myself
		if other==person:
			continue
		sim=similarity(prefs,person,other)

		# ignore scores of zero or lower
		if sim<=0.5: 														#set  a threshold
			continue

		for item in prefs[other]:
			# only score movies I haven't seen yet
			if item not in prefs[person]:								# or prefs[person][item]==0:
				# Similarity * Score
				totals.setdefault(item,0)
				totals[item]+=prefs[other][item]*sim
				# Sum of similarities
				simSums.setdefault(item,0)
				simSums[item]+=sim
				# Create the normalized list

	rankings=[(total/simSums[item],item) for item,total in totals.items()]
	# Return the sorted list
	rankings.sort()
	rankings.reverse()

	return rankings


def loadMovieLens(path='/home/abhishek/Desktop/miniproject_presentation/MatrixFactorization_crossdomain/again_data_divide', file='train.txt'):
	# Load data
	prefs={}
	for user in range(6041) :
		prefs.setdefault(str(user),{})

	for line in open(path+file):
		(user,movieid,rating)=line.split(':')
		prefs[user][movieid]=float(rating)
	return prefs


if __name__=='__main__':
	file=open("Test_CF.dat","a")	
	tmp=10
	while tmp<=100:
		
		trainPrefs = loadMovieLens(file="/d"+str(tmp)+"Train.dat")
		testPrefs = loadMovieLens(file="/d"+str(tmp)+"Test.dat")
		
		final_MAE=[]
		final_RMSD=[]
		notPredicted=0
		true_positive=0
		total=0											#for calculating recall
		for user in testPrefs:	
			pred = getRecommendations(trainPrefs,user)
			preds={}
			for rating,item in pred:
				preds[item]=rating
				# print movies[item],rating,item
			
			for movie in testPrefs[user]:
				total+=1
				if not movie in preds:
					notPredicted+=1
					continue 
				actualRating = testPrefs[user][movie]
				predcitedRating = preds[movie]
				#precision calculation

				#liking
				if predcitedRating>=2.5 and actualRating>=2.5:
					true_positive+=1
				#disliking	
				elif predcitedRating<=2.5 and actualRating<=2.5:
					true_positive+=1

				MAE = fabs(predcitedRating - actualRating)
				RMSD = MAE*MAE
				#print (predcitedRating,actualRating,diff)
				final_MAE.append(MAE)
				final_RMSD.append(RMSD)
				#if len(final_MAE)!=0 :
					#print ("MAE=%lf, RMSD=%lf" % (sum(final_MAE)/len(final_MAE), sum(final_RMSD)/len(final_RMSD)))

		FMAE=(sum(final_MAE)/len(final_MAE))
		FRMSE=sqrt(sum(final_RMSD)/len(final_RMSD))
		PRECISION=(1.0*true_positive/total)*100
		#print(total,notPredicted,true_positive)
		RECALL=(1.0*(total-notPredicted)/total)*100	
						
		print ("%d final MAE=%lf, RMSD=%lf %lf %lf" % (tmp,FMAE,FRMSE,PRECISION,RECALL))
		file.write("%d %lf %lf %lf %lf\n" %(tmp,FMAE,FRMSE,PRECISION,RECALL))
		tmp=tmp+10
	file.close();

#MAE=0.989619, RMSD=1.707827 (sim_threshold=0.5)
