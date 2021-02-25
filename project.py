from threading import *
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import time
from datetime import datetime
import statistics
import math

# This class holds a user object. Where each user has the following:
# userId: Is the specific Id for the current user.
# ratingsList: Is the list of previous ratings the user has given
# dateList: is the list of dates the previous user rated on.
# movieIdList: is the list of movies the user has rated.
# *The index for ratingList, dateList, and movieIdList are correlated where
# userId rated <movieIdList[i]> on date <dateList[i]> and gave a rating of <ratingList[i]>
class User:
    def __init__(self, userId, rating, date, movieid):
        # the given values are linked by their index for example if you want to know the rating and date of the movie in index 2 in movieIdList had then
        # look at index 2 of the ratingsList and dateList
        self.userId = userId
        self.ratingList = [rating]
        self.dateList = [date]
        self.movieIdList = [movieid]

    def addFields(self, rating, date, movieid):
        self.ratingList.append(rating)
        self.dateList.append(date)
        self.movieIdList.append(movieid)


# This method reads in the data from the files.
# filename: is the file to read data from
# dataset: is the 2d list that will hold the file data each inner array holds a row from the file.
# lock: is used in multithreading to make sure only 1 thread has access to datastructure at each moment.
def read(fileName, dataset, lock):
    with open(fileName, encoding="utf8") as fp:
        # keeps track of the total amount of movies read in
        movieCount = 0

        # goes through each line of the file
        for line in fp:
            # if line read in ends with : (indicates it is a line containing the movie id)
            if line[-2] == ":":
                movieCount = movieCount + 1
                movieid = int(line.split(":")[0])
                # Will stop reading data from file if movies read in is above MAX_MOVIE_COUNT or MAX_MOVIE_COUNT = -1(indicates to read entire file)
                if MAX_MOVIE_COUNT != -1 and movieCount > MAX_MOVIE_COUNT:
                    break
            # else line in text file holds user information
            else:
                # processing line
                rowVal = line.split(",")
                userId = rowVal[0].strip()
                rating = int(rowVal[1].strip())
                date = rowVal[2].strip()
                d = datetime.strptime(date, "%Y-%m-%d")
                date = d.timestamp() / (60 * 60 * 24)

                # adding line to datastructures
                lock.acquire()
                dataset.append([userId, rating, date, movieid])
                lock.release()
    fp.close()


# This method helps the preProcess() method by computing the adjRating, stdRating, adjDate, and stdDate for the user passed in.
# Returns: a list containing the specified users [adjRating, stdRating, adjDate, and stdDate]
def preProcessHelper(user):
    userId = user.userId
    adjRating = statistics.median(user.ratingList)
    adjDate = statistics.median(user.dateList)
    stdDate = 0
    stdRating = 0
    if len(user.ratingList) > 1:
        stdDate = statistics.stdev(user.dateList)
    if len(user.dateList) > 1:
        stdRating = statistics.stdev(user.ratingList)
    return [userId, adjRating, stdRating, adjDate, stdDate]


# This method preprocceses the training data by taking all users in the userlist and computing the
# adjRating: median of the users previous ratings
# stdRating: standard deviation of the users previous ratings
# adjDate: median of the users previous dates for ratings
# stdDate: standard deviation of the users previous dates for ratings
# Returns: a list of users with adjRating, stdRating, adjDate, and stdDate
def preProcess(userList):
    preprocessedList = []
    # for each user calculates the median and std of rating and date
    for user in userList.values():
        preprocessedList.append(preProcessHelper(user))
    return preprocessedList


# This method takes two lists testList and acutalList and compares element by element, returning the number of correct predictions
# testList: list of predicted classification values for the test dataset.
# actualList: the list of acutual classification values for the test datset.
# Returns: the total amount of correct predictions between testlist and actualList.
def accuracy(testList, actualList):
    if len(testList) == len(actualList):
        numCorrect = 0
        for idx in range(0, len(testList)):
            if testList[idx] == actualList[idx][0]:
                numCorrect = numCorrect + 1
        return numCorrect
    else:
        return -1


# This method takes in the training dataset and fills the userList and movieList with its values.
# userList: holds a list of all the movies, ratings, and dates given by that user
# movieList:  gives a list of all the movie with key:movieid, value: list of userids who rated given movie
# splitXTrain: gives the X_training values - [userId, date, movieId]
# splitYTrain: gives the Y_training values -[rating]
# lock: used for multithreading to prevent multiple threads from modifiying above datastrucutres at the same time.
def fillList(userList, movieList, splitXTrain, splitYTrain, lock):
    tempXTrain = X_train.tolist()
    tempYTrain = Y_train.tolist()
    for i in range(0, len(tempXTrain)):
        userId = tempXTrain[i][0]
        date = tempXTrain[i][1]
        movieId = tempXTrain[i][2]
        rating = tempYTrain[i][0]

        # add to movieList
        if movieId in movieList:
            lock.acquire()
            movieList.get(movieId).append(int(userId))
            lock.release()
        else:
            lock.acquire()
            movieList[movieId] = [int(userId)]
            lock.release()

        # add to userList
        if userId in userList:
            lock.acquire()
            user = userList.get(userId)
            user.addFields(rating, date, movieId)
            lock.release()
        else:
            user = User(userId, rating, date, movieId)
            lock.acquire()
            userList[userId] = user
            lock.release()


# This method takes list of closest users to the current user and removes all users that haven't rated the specified movieId
# neighborList: Is an array that holds a list of nearest users who rated the most similair to the given user this list belongs to.
# movieListVals: Is the list of users who rated a given movieId.
# KNN: gives the amount of neighbors you want.
# Return: this method returns the list of nearest users who rated the most similair to the given user this list belongs to who also rated the specified movie we are trying to find.
def movieIdUsers(neighborList, movieListVals, KNN):
    count = 0
    tempReturn = []
    for neighbor in neighborList:
        if count == KNN:
            break
        if neighbor in movieListVals:
            tempReturn.append(neighbor)
            count = count + 1
    return tempReturn


# This method will go through the testing list and make predictions on the ratings each row would give using the model.
# userList: holds a list of the users of the training dataset
# movieList : holds a list of all the movies rated in the training dataset along with the users who rated each specifiec movie.
# X_test: Gives the explanatory variables for the testing dataset [userId, date, movieId]
# Y_test: Gives the categorization variable for the testing dataset [rating]
# preprocessDf: Gives a list of users and their preprocessed scores [adjRating, stdRating, adjDate, stdDate]
# model: gives the KNN model that we trained the data on
# numCorrect: gives a list that holds the number of correct predicitions for each thread.
# lock: used for multithreading to prevent multiple threads from accessing the same variable at the same time.
def testing(userList, movieList, X_test, Y_test, preprocessDf, model, numCorrect, lock):
    predRating = []
    X_testList = X_test.tolist()
    # Goes through each value in X_test
    for i in range(len(X_test)):
        userId = X_testList[i][0]
        date = X_testList[i][1]
        movieId = X_testList[i][2]
        # ---------------------STEP 5: PREPROCESS TESTING -----------------------
        # to run our test values into the model we need to change the X_test data from [userId, date, movieId] into the proper form which is [adjRating, stdRating, adjDate, stdDate](This gives a generalized pattern of the users previous ratings and date by looking at the average and spread of the rating and date variables.)
        # if the userid in the X_test row was in the training set(specific user id was trained on in the model) then it means that the userId for the row has prior values it can predict off of.
        # in this case use the preprocessed user's data for the given user as the explanatory variable for testing.
        if userId in userList:
            adjUserVal = preprocessDf.loc[
                preprocessDf["UserId"] == userId
            ].values.tolist()[0][1:]
        # if the userid wasn't found to be trained on by the training data then there is not previous rating information that can be used to compare with each other user
        # in this case use the explanatory variable [adjRating = 3, stdRating = 2, adjDate = X rows date, stdDate =0] this makes a user who has a rating pattern of having
        # equal change of choosing all possible ratings (1-5) but rates soley on given date.
        else:
            # if user isn't found in previous ratings then make user rating median to 3 and std to 2(basically within 1 std user has chance to rate from 1 - 5 which is basically random guesing)
            # try to predict user off date user gave info.
            adjUserVal = [3, 2, date, 0]

        # ----------------------------STEP 6: MODEL TESTING---------------------------
        # The preprocessed testing data row gets passsed into the model and a list of all the neighbors from nearest to farthest in pattern similairty to the current user is given.
        adjXList = np.asarray(adjUserVal).reshape(-1, 4)
        nearestNeighbors = model.kneighbors(
            X=adjXList,
            n_neighbors=math.ceil((len(X_train) - 1) / 2),
            return_distance=False,
        ).tolist()

        neighborList = nearestNeighbors[0]
        tempProcess = []

        # This line goes through the nearest neighbors and removes all the neighbors who haven't rated the given movieId found in the X test line.
        if movieId in movieList:
            movieListVals = movieList[movieId]
            tempProcess = movieIdUsers(neighborList, movieListVals, KNN)

        # -----------------------------STEP 7 : PREDICTION -----------------------------------------
        # No nearest neighbors who rated given movie we need to predict using previous ratings of user using previous rating score
        if len(tempProcess) == 0:
            predRating.append(adjUserVal[0])
        # If there are nearest neighbors who rated given movie for the current X_test user then find the supermajority of the ratings and predict that value.
        else:
            # find supermajority of ratings and predict that
            user = userList.get(str(tempProcess[0]))
            movieIdx = user.movieIdList.index(movieId)
            rating = user.ratingList[movieIdx]
            predRating.append(rating)
    lock.acquire()
    # Calculates the number of correct predictions from the estimated ratings in the classification and the actual ratings.
    acc = accuracy(predRating, Y_test.tolist())
    numCorrect.append(acc)
    lock.release()


if __name__ == "__main__":

    t1 = time.time()

    # -------STEP0: READ DATSET --------------

    # gives a 2d array of the entire dataset
    dataset = []

    # used for multithreading
    lock = Lock()
    # holds a list of the currently running threads
    threads = []

    # used for debugging; reads only a subset of the data by limiting the max amount of movies read per file
    # if you want all the data to be read change maxMovie count to -1
    MAX_MOVIE_COUNT = -1
    # number of neighbors to choose from
    KNN = 1
    # goes through the 4 combined_data.txt files and reads through them
    for i in range(1, 5):
        filename = "data/combined_data_" + str(i) + ".txt"
        # creates a thread for each combined_data file - each thread executes the read method with the arguments specified in the arugment field below
        thread = Thread(target=read, args=(filename, dataset, lock,),)
        threads.append(thread)
        thread.start()

    # waits for all the threads to finish
    for thread in threads:
        thread.join()
    # --------------STEP 1: SPLIT DATA----------------------
    model = neighbors.KNeighborsClassifier(n_neighbors=5)
    kf = StratifiedKFold(n_splits=10, shuffle=True)
    accuracyList = []

    df = pd.DataFrame(dataset, columns=["UserId", "Rating", "Date", "MovieId"],)
    XList = df[["UserId", "Date", "MovieId"]].values
    YList = df[["Rating"]].values

    # Splits data into train and test sets with specified number of cross validation
    for train_index, test_index in kf.split(XList, YList):
        X_train, X_test, Y_train, Y_test = (
            XList[train_index],
            XList[test_index],
            YList[train_index],
            YList[test_index],
        )

        # gives a dictionary with key : userid ,value : User Object
        # User Object: holds a list of all the movies, ratings, and dates given by that user
        userList = {}
        # gives a list of all the movie with key:movieid, value: list of userids who rated given movie
        movieList = {}

        # Creates threads to go through the training data and fill the userList and movieList
        threads = []
        NUM_THREADS = 4
        splitXTrain = np.array_split(X_train, NUM_THREADS)
        splitYTrain = np.array_split(Y_train, NUM_THREADS)
        for i in range(NUM_THREADS):
            thread = Thread(
                target=fillList,
                args=(userList, movieList, splitXTrain[i], splitYTrain[i], lock,),
            )
            threads.append(thread)
            thread.start()

        # waits for all the threads to finish
        for thread in threads:
            thread.join()

        # -----------STEP 2: PREPROCESSING TRAINING-----------------
        # Preprocceses the training data by taking all users in the userlist and computing the
        # adjRating: median of the users previous ratings
        # stdRating: standard deviation of the users previous ratings
        # adjDate: median of the users previous dates for ratings
        # stdDate: standard deviation of the users previous dates for ratings

        preprocessedList = preProcess(userList)
        preprocessDf = pd.DataFrame(
            preprocessedList,
            columns=["UserId", "adjRating", "stdRating", "adjDate", "stdDate"],
        )
        XPreProcesslist = preprocessDf[
            ["adjRating", "stdRating", "adjDate", "stdDate"]
        ].values
        YPreProcesslist = preprocessDf["UserId"].values
        # -----------STEP 3: MODEL SELECTION-----------------
        # creates a model using the preprocessed training data where categorization variable is "userId" and explanatory variables are [adjRating, stdRating, adjDate, stdDate]
        # this gives a model that will predict the users who are most similair in rating patterns to the current user given the current users [adjRating, stdRating, adjDate, stdDate]
        model.fit(XPreProcesslist, YPreProcesslist)

        # -----------STEP4: TESTING ----------------
        # This section goes through each of the X_testing data and predicts the rating using the model. Multithreading is used to speed up this process.
        # More information is found in the testing method about how testing occurs.
        NUM_THREADS = 1000
        numCorrect = []
        totalTestLen = len(X_test)
        splitXTest = np.array_split(X_test, NUM_THREADS)
        splitYTest = np.array_split(Y_test, NUM_THREADS)
        threads = []
        for i in range(NUM_THREADS):
            thread = Thread(
                target=testing,
                args=(
                    userList,
                    movieList,
                    splitXTest[i],
                    splitYTest[i],
                    preprocessDf,
                    model,
                    numCorrect,
                    lock,
                ),
            )
            threads.append(thread)
            thread.start()

        # waits for all the threads to finish
        for thread in threads:
            thread.join()
        # -------------STEP 8: ACCURACY------------------------
        # *Steps 5-7 are found in the testing method.
        # This section computes the overall accuracy in the current K fold split by taking the total correct predictions/ total datas predicted.
        acc = sum(numCorrect) / totalTestLen
        accuracyList.append(sum(numCorrect) / totalTestLen)

    runTime = time.time() - t1
    runTimeMin = runTime / 60
    print("Run time Minutes: " + str(runTimeMin))
    print("List of accuracies for each Cross Validation: " + str(accuracyList))
    print("Total Accuracy: " + str(statistics.mean(accuracyList)))
    print("Done")
