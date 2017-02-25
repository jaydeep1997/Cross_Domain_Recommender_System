from math import sqrt

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


# for person1 in critics:
# 	for person2 in critics:
# 		#if person1!=person2:
# 			print person1+"  "+person2+":",
# 			print sim_distance(critics,person1,person2),
# 			print sim_pearson(critics,person1,person2)


def top_matches(critics, person, n=5, sim=sim_pearson):
	li=[(sim(critics,person,other),other) for other in critics if other!=person]

	li.sort()
	li.reverse()

	return li[0:n]				#return only first n items

def transformPrefs(prefs):
	result={}
	for person in prefs:
		for item in prefs[person]:
			result.setdefault(item,{})

			# Flip item and person
			result[item][person]=prefs[person][item]
	
	return result



# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(domain1,domain2,person,similarity=sim_pearson):

	totals={}
	simSums={}

	for other in domain2:
		# don't compare me to myself
		if other==person:
			continue
		sim=similarity(domain2,person,other)

		# ignore scores of zero or lower
		if sim<=0: 
			continue

		for item in domain1[other]:
			# only score movies I haven't seen yet
			if item not in domain1[person]:								# or prefs[person][item]==0:
				# Similarity * Score
				totals.setdefault(item,0)
				totals[item]+=domain1[other][item]*sim
				# Sum of similarities
				simSums.setdefault(item,0)
				simSums[item]+=sim
				# Create the normalized list

	rankings=[(total/simSums[item],item) for item,total in totals.items()]
	# Return the sorted list
	rankings.sort()
	rankings.reverse()

	return rankings

#NOTE: here prefs[user_id][movie_id]
def loadMovieLens(path='/home/jaydeep/Desktop/Mini_project/movielens_dataset/movielens_100K',file='/u1.base'):
	# Get movie titles
	movies={}
	for line in open(path+'/u.item',encoding='latin-1'):
		(id,title)=line.split('|')[0:2]
		movies[id]=title

	# Load data
	prefs={}
	for line in open(path+file):
		(user,movieid,rating,ts)=line.split('\t')
		prefs.setdefault(user,{})
		prefs[user][movieid]=float(rating)
	return prefs

prefs1=loadMovieLens(file='/u1.base')
prefs2=loadMovieLens(file='/u2.base')

print (getRecommendations(prefs1,prefs2,'1',sim_pearson))