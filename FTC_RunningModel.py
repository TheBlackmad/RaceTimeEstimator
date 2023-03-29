# TheBlackmad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import pickle
import datetime
from datetime import timedelta, timezone
import numpy as np
import pandas as pd
from math import sqrt, floor, ceil

# Machine Learning and Deep Learning Libraries
import sklearn.model_selection
from sklearn import linear_model as lm
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

import FTC_GPX

class RunningModel:
    '''
        This class represents a model of the runner based upon the workouts feed (gpx files)
    '''

    def __init__(self, folderpath):
        '''
            This routine initialize the RunningModel object what models a runner based upon his GPX workouts. It reads
            all the workouts and load this object with this data. There are several models included:

            - Riegel: Riegel's formula for calculating races time.
            - Daniel: Daniel's formula considering slopes.
            - Model: based upon regression models.

            Args:
                folderpath: GPX folderpath for the GPX

            Returns:
                self

            Raises:
                Exception: None
        '''
        self.correctionModel = None         # This is the correction model params for Daniels
        self.folderpath = folderpath        # location of the GPX files
        self.regression_model = []          # regression models for each slope category
        self.workouts = []                  # each individual workout
        self.__slopePace = []               # this creates slopes and paces raw
        GPXFilenames = []

        # ensure the path ends with "/"
        if not folderpath.endswith("/"):
            folderpath = folderpath + "/"

        # retrieve the list of GPX files
        for filename in os.listdir(folderpath):
            if filename.endswith(".gpx") or filename.endswith(".GPX"):
                GPXFilenames.append(folderpath + filename)

        print("List of GPX Files: ", GPXFilenames)
        # For each filename we create the workouts
        total = len(GPXFilenames)
        print("Loading the list of GPX files . . .")
        for itemNumber, filename in enumerate(GPXFilenames):
            self.addNewWorkout(filename)
            print(itemNumber+1, " / ", total, " - ", filename)

        print("All GPX Files were loaded !")

    # add new workout to the model
    def addNewWorkout(self, filename):
        '''
            This routine adds a new workout to the model.

            Args:
                filename: GPX file

            Returns:
                None

            Raises:
                None
        '''

        # read GPX File. if it exists the object .oftc reads from it. It is a much quicker access.
        fname = filename.replace('.gpx', '.oftc')
        print(f"Checking file {fname}")
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                gpx = pickle.load(f)
        else:
            print(f"Reading file {filename}")
            gpx = FTC_GPX(filename)
            with open(fname, 'wb') as f:
                pickle.dump(gpx, f)

        # include in the list of workouts: time, distance, date, ...
        # be carefull with the GAP values, as we need to take out the outliers (use clean data)
        df_filtered = self.__cleanWorkout(gpx.opt_route_df)

        gap = df_filtered['GAP'].mean()
        hr = gpx.opt_route_df['hr'].mean()
        ef=0.0 if hr<=0 else ( (60/gap)*(1000/60) ) / hr

        self.workouts.append({
            'date': gpx.getDate(),
            'distance': gpx.opt_route_df['dist_vin'].max(),
            'duration': gpx.getElapsedTime(),
            'GAP': gap,
            'HR': hr,
            'EfficiencyFactor': ef
        })

        # Add Slope and Pace into the Slope and Pace List
        self.__updateSlopePaces(df_filtered, (gpx.getDate() - datetime.datetime.now(timezone.utc)).days)

        print(f"Adding {round(gpx.opt_route_df['dist_vin'].max()/1000, 2)} kms in {gpx.getElapsedTime()} on {gpx.getDate()} with a GAP of {round(gap, 2)} min/km with elevation of {round(gpx.getUphillDownhill()[0])} mts. Average Heart Rate: {round(hr, 2)} and Efficiency Factor: {round(ef, 2)}")

    # Method that just creates relations between slopes and paces
    def __updateSlopePaces(self, df, daysAgo, periodInDays=365):
        '''
            This routine updates the internal state of the slope/pace list, including the days ago when this relation was created.
            weights are not currently used and will probably be removed.

            Args:
                df: dataframe with the data
                daysAgo: when the workout was performed
                periodInDays: to weighten the slope/pace

            Returns:
                None

            Raises:
                None
        '''

        # include in the list of slopes
        for index, each_row in df.iterrows():
            self.__slopePace = self.__slopePace + [{
                'days_ago': daysAgo,
                'slope': each_row['slope'],
                'pace': each_row['pace'],
                'weight': 1 + daysAgo / periodInDays  # assuming training affecting this
            }]

    def __cleanWorkout(self, df, paceMax=20.0, paceMin=3.0, slopeMax=0.60, slopeMin=-0.60, qMax=0.95, qMin=0.05):
        '''
            This routine cleans the Slope Pace list from the model. The cleaning consist of removing outliers from
            pace and slopes, and keeping a quantile of samples.

            Args:
                dfInput: Dataframe with the Workout to clean
                paceMax, paceMin: interval for the pace to clean
                slopeMax, slopeMin: interval for the pace to clean
                qMax, qMin: interval for the quantile

            Returns:
                clean workout data (DataFrame)

            Raises:
                None
        '''

        # To consider it valid, pace should be between 20.0 and 3.0
        # To consider it valid, slope should be between 0
        # To consider it valid, the slope should be performed not more than periodInDays days ago
        df_filtered = df[(df['pace'] <= paceMax) & (df['pace'] > paceMin) & (df['slope'] < slopeMax) & (df['slope'] > slopeMin) & (df['slope'] != 0.0)]
#        df = df[(df['days_ago'] >= -periodInDays)]

        # to make it consistent, we do not consider extreme values from th whole samples
        # Remove values outside quantiles 0.05 - 0.95
        pace_q_low = df_filtered["pace"].quantile(qMin)
        pace_q_hi = df_filtered["pace"].quantile(qMax)
        df_filtered = df_filtered[(df_filtered["pace"] < pace_q_hi) & (df_filtered["pace"] > pace_q_low)]

        return df_filtered

    # Apply Riegel formula
    def Riegel(self, newRaceDistance, oldRaceDistance, oldTime):
        '''
            This routine provides the time for the new race distance based upon an existing race time and distance.

            Args:
                newRaceDistance: race distance to calculate time for
                oldRaceDistance: known race distance for which we have a time
                oldTime: time spent on the known race distance

            Returns:
                time for the race distance

            Raises:
                Exception: Assert oldRaceDistance is not a number > 0
        '''
        assert oldRaceDistance > 0
        return oldTime*(newRaceDistance/oldRaceDistance)**1.06

    # Apply Daniels formula
    def Daniel(self, distances, slopes, secsPerGradeplusPerKm=9.32, secsPerGrademinusPerKm=-4.97, StopFactor=1.125):
        '''
            This routine the Daniel estimation for a race.
            Source: https://www.livestrong.com/article/533573-running-up-hills-vs-flat-time/

            Args:
                distances: race distance parts in meters. Sum of it all is the length of the race
                slopes: slopes for each distance part in 0..1
                secsPerGradeplusPerKm: Daniel modifiers for positive slopes, additional time in seconds per 1% per kms
                secsPerGrademinusPerKm: Daniel modifiers for negative slopes, less time in seconds per 1% per kms
                StopFactor: factor considering non moving time

            Returns:
                Daniel's time for the race distance

            Raises:
                Exception: Assert oldRaceDistance is not a number > 0
        '''
        addTimes = []
        addTimes.clear()

        for i in range(len(distances)):
            s = (slopes[i]) * 100

            # There are two options in theory: 1. use Daniels formula directly, or 2. use the Grade Adjustment
            # Grade Adjustment formula is: 0.0021 * s ** 2 + 0.034 * s + 1
            # Daniels formula uses modifiers for the pace per Km per Grade*100
            if s > 0.0:
                addTimes.append(secsPerGradeplusPerKm * s * distances[i] / 1000)
            else:
                addTimes.append(secsPerGrademinusPerKm * s * distances[i] / 1000)

        total_time = (sum(addTimes) / 60 + self.refGap * sum(distances) / 1000) * StopFactor
        return total_time

    # Apply Daniels formula
    def GradeAdjustment(self, distances, slopes, StopFactor=1.125):
        '''
            This routine the Daniel estimation for a race.
            Source: https://www.livestrong.com/article/533573-running-up-hills-vs-flat-time/

            Args:
                distances: race distance parts in meters. Sum of it all is the length of the race
                slopes: slopes for each distance part in 0..1
                secsPerGradeplusPerKm: Daniel modifiers for positive slopes, additional time in seconds per 1% per kms
                secsPerGrademinusPerKm: Daniel modifiers for negative slopes, less time in seconds per 1% per kms
                StopFactor: factor considering non moving time

            Returns:
                Daniel's time for the race distance

            Raises:
                Exception: Assert oldRaceDistance is not a number > 0
        '''
        addTimes = []
        addTimes.clear()

        for i in range(len(distances)):
            s = (slopes[i]) * 100

            # There are two options in theory: 1. use Daniels formula directly, or 2. use the Grade Adjustment
            # Grade Adjustment formula is: 0.0021 * s ** 2 + 0.034 * s + 1
            # Daniels formula uses modifiers for the pace per Km per Grade*100
            t = 0.0021 * s ** 2 + 0.034 * s + 1
            k = self.refGap * (t-1)
            addTimes.append(k)

        total_time = (sum(addTimes) / 60 + self.refGap * sum(distances) / 1000) * StopFactor
        return total_time

    # Method to train the model by using regression
    def ModelFit(self, periodInDays=365):
        '''
            This routine train the model based upon the workouts added to this model. The model is a regression
            machine learning model. The models for each individual slope are included in the self.regression_model

            Args:
                periodInDays: is used for the Mean Calculation only considering data within the last periodInDays days

            Returns:
                None

            Raises:
                Exception: Assert oldRaceDistance is not a number > 0
        '''

        X_list = []
        Y_list = []

        # creating the array of data for the model
        X_list.clear()
        Y_list.clear()

        # for the model fit use only data not older than periodInDays days
        df = pd.DataFrame(self.__slopePace)
        df = df[(df['days_ago'] >= -periodInDays)]

        step = 5
        minmax = self._getSlopeIntervals(df)
        #        print(f"minmax: {minmax}")
        for i in range(len(minmax)):
            #        while i < len(minmax):
            min = minmax[i]['min']
            max = minmax[i]['max']

            df2 = df[(df['slope'] >= min) & (df['slope'] < max)]

            for index, each_row in df2.iterrows():
                X_list.append((
                    float(each_row['days_ago']),
                    float(each_row['slope']),
                ))
                Y_list.append((
                    float(each_row['pace'])
                ))

            X = np.array(X_list)
            Y = np.array(Y_list)

            # splitting the data into training and testing
            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, shuffle=True,
                                                                                        test_size=0.2)

            # Predicting the model as a Linear Regression
            #            paceModel = lm.LinearRegression()
            paceModel = make_pipeline(SplineTransformer(n_knots=2, degree=5), lm.Ridge(alpha=0.09))

            #            print("Performing Training")
            paceModel.fit(X_train, Y_train)
            #            print(f"*****\nSlope interval [{min}, {max}]")
            #            print("Coeficients: ", paceModel.coef_)
            #            print("Intercept: ", paceModel.intercept_)
            #            print("Score: ", paceModel.score(X, Y))

            # Predicting new values
            #            predictions = paceModel.predict(X_test)
            #            r2 = r2_score(Y_test, predictions)
            #            rmse = mean_squared_error(Y_test, predictions, squared=False)

            #            print(f"The r2 is: {r2}")
            #            print(f"The rmse is: {rmse}")

            self.regression_model.append({
                'min': min,
                'max': max,
                'model': paceModel,
            })

    # The model predicts race time using Machine Learning
    def ModelPredict(self, distances, slopes, elevation, stopFactor=1.125, periodInDays=365):
        '''
            This routine predicts the time using statistic for the given race track. The race track is given by the
            distances and slopes.

            Args:
                distances: race distance parts in meters. Sum of it all is the length of the race
                slopes: slopes for each distance part in 0..1
                elevation: elevation at which the race track starts
                stopFactor: factor considering non moving time
                periodInDays: is used for the Mean Calculation only considering data within the last periodInDays days

            Returns:
                Daniel's time for the race distance

            Raises:
                Exception: Assert oldRaceDistance is not a number > 0
        '''

        # data for the slopes is clean. Just keep data within the time limit.
        df = pd.DataFrame(self.__slopePace)
        df = df[(df['days_ago'] >= -periodInDays)]

        time_total = 0.0
        time_total_model = 0.0
        currElevation = elevation
        corr = []
        for i in range(len(distances)):
            trackTimeMean = (distances[i] / 1000) * self._getAvgPace(df, slopes[i])
            trackTimeModel = (distances[i] / 1000) * (self.__PaceModelPredict(0, slopes[i]))
            time_total += trackTimeMean
            time_total_model += trackTimeModel
            currElevation += distances[i] * slopes[i]
            correction_time = self._getAvgPace(df, slopes[i]) - self.refGap
            #            print(f"SLOPE: {slopes[i]} - CORRECTION: {round(60*correction_time)}sec/km / {round(60*correction_time/abs(slopes[i]*100))}sec/km per 1%")
            #            print(f"Calculation from getPace: {self.getPace(df2, slopes[i])} \t Calculation from Regression: {self.FernandoCorrectionModelPredict(-0, slopes[i])}")
            #            print(f"Calculation from getPace: {df2['pace'].mean()} \t Calculation from Regression: {self.FernandoCorrectionModelPredict(-150, slope)}")

            corr.append({
                'slope': slopes[i],  # between 0..1
                'distance': distances[i],  # in mts
                'elevation': currElevation,
                'Daniels correction': 60 * correction_time / abs(slopes[i] * 100) if slopes[i] != 0 else 0.0,
                # in secs per km per 1%
                'Pace by Mean': self._getAvgPace(df, slopes[i]),  # in min/km
                'Time by Mean': trackTimeMean,  # in min total time for the track
                'Pace by Regression': self.__PaceModelPredict(0, slopes[i]),  # in min/km
                'Time by Regression': trackTimeModel,  # in min total time for the track
                'Stop Factor': stopFactor
            })

        return corr, time_total * stopFactor, time_total_model * stopFactor  # for the time at the stops

    # The model predicts pace for a specific slope
    def __PaceModelPredict(self, days_ago, slope):
        '''
            This routine predicts the pace to follow for a given slope.

            Args:
                slopes: slopes to calculate pace from
                days_ago: (unused) estimate pace as per days_ago days ago

            Returns:
                pace

            Raises:
                None
        '''
        X_list = []

        # Predict
        X_list.clear()
        X_list.append((
            days_ago, slope
        ))
        X = np.array(X_list)

        # Lets apply regression based on the parameters
        for i in range(len(self.regression_model)):
            if (slope > self.regression_model[i]['min']) & (slope <= self.regression_model[i]['max']):
                ret = self.regression_model[i]['model'].predict(X)
                return ret[0]

        print (f"ERROR while applying model predict! days_ago={days_ago} slope={slope}")

    def _getAvgPace(self, df, slope, step=0.05):
        '''
            This routine provides the average pace at a specific slope

            Args:
                df: dataframe to use samples from.
                slope: given slope to calculate average pace
                step: step considered for the slope interval

            Returns:
                average pace

            Raises:
                None
        '''

        if slope > 0.0:
            min = floor(slope / step) * step
            max = min + step
        else:
            max = ceil(slope / step) * step
            min = max - step

        df2 = df[(df['slope'] >= min) & (df['slope'] < max)]
#        print(f"For slope {slope} we need to account for {df2['pace'].mean()-self.refGap} secs more per 1% and km")
        if len(df2) == 0:
            return 10.0         # default pace in the mountain

        return df2['pace'].mean()

    def _getNumberSamples(self, df, i, j):
        '''
            This routine counts the number of samples that the slope interval between i and j has

            Args:
                df: dataframe to use samples from.
                i: lowest slope
                j: maximum slope

            Returns:
                number of samples

            Raises:
                None
        '''
        # Number of Samples for these limits (min, max)
        df2 = df[(df['slope'] >= i) & (df['slope'] < j)]
        pace_q_low = df2["pace"].quantile(0.05)
        pace_q_hi = df2["pace"].quantile(0.95)
        df_filtered = df2[(df2["pace"] < pace_q_hi) & (df2["pace"] > pace_q_low)]
        return len(df_filtered)

    def _getSlopeIntervals (self, df, step0=0.025, minSamples=10):
        '''
            This routine provides a set of intervals from slope min to max with the number of samples for each
            interval

            Args:
                df: dataframe to use samples from.
                step0: each interval should be of the same size
                minSamples: minimum number of samples that each interval must have

            Returns:
                minmax intervals with the number of samples for each one

            Raises:
                None
        '''

        minmax = []
        step = step0
        min = -1
        max = 1
        i = min

        # generate the intervals
        while i < max:
            nSamples = 0
            while nSamples < minSamples:
                i = i + step
                nSamples = self._getNumberSamples(df, min, i)
                if i >= 100:                    # this is the condition while the slope is lower than 100%
                    nSamples = 999999           # this value indicates that we need to get the same data as in the previous interval
                    i = 100
            minmax.append({
                'min': min,
                'max': i,
                'samples': nSamples
            })
            min = i

        return minmax

    # Estimates pace in a given race
    def estimatePaceInRace (self, raceDistance, weightDistance=0.5, weightRecently=0.5, periodInDays=365, selection=0.2):
        '''
            This routine predicts the pace for a race given the distance. gets raceDistance to calculate possible
            pace in race. it calculates based upon recent races (more recent means bigger weights in calculations)
            and the distance of the race by similarity (similar races in distance gets more weight)
            by default they are set to 50% each weight.

            Args:
                raceDistance: to calculate possible pace in race
                weightDistance: weight to similar distance runs
                weightRecently: weight to recent runs
                periodInDays: contains as well the timeline where to consider runs.
                select: selects the percentage of workouts more similar to them to calculate the average pace.

            Returns:
                pace

            Raises:
                None
        '''
        lst = []
        lst.clear()

        # Create the list of runs/workouts
#        nWorkouts = len(self.source.GPXList)
        nWorkouts = len(self.workouts)
        for workout in self.workouts:
            daysAgo = (workout['date'] - datetime.datetime.now(timezone.utc)).days
            distance = workout['distance']
            duration = workout['duration']
            gap = workout['GAP']
            distanceDiff = 1-((distance-raceDistance)/raceDistance)**2
            currentRate = 1+daysAgo/periodInDays
            affinity = weightDistance*distanceDiff + weightRecently*currentRate
#            print(f"{daysAgo} | {distance} | {duration} | {gap} | {distanceDiff} | {currentRate} | {affinity}")
            lst.append({
                'Days_Ago': daysAgo,
                'Distance': distance,
                'Duration': duration,
                'GAP': gap,
                'Distance_Diff': distanceDiff,
                'Current_Rate': currentRate,
                'Affinity': affinity
            })

        # . . . OR THIS CAN BE AS WELL A REGRESSION BASED ON DISTANCE AND DAYS_AGO
        # WHERE TO ESTIMATE THE GAP
        X_list = []
        Y_list = []

        for each_row in lst:
            X_list.append((
                float(each_row['Days_Ago']),
                float(each_row['Distance']),
            ))
            Y_list.append((
                float(each_row['GAP'])
            ))
#            print(f"@({each_row['Days_Ago']}, {each_row['Distance']}) = {each_row['GAP']}")

        X = np.array(X_list)
        Y = np.array(Y_list)

        # splitting the data into training and testing and
        # Predicting the model as a Linear Regression
        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, shuffle=True, test_size=0.2)
        paceModel = make_pipeline(SplineTransformer(n_knots=2, degree=5), lm.Ridge(alpha=0.09))
        paceModel.fit(X_train, Y_train)

        # Predict GAP for distance
        X_list.clear()
        X_list.append((0, raceDistance))
        X = np.array(X_list)
        ret = paceModel.predict(X)
        self.refGap = ret[0]
        self.refDistance = raceDistance
#        print(f"It has been predicted that for {round(raceDistance/1000,2)} km, the GAP would be {self.timeFromMinutes(round(ret[0],2))} mins/km")
        return ret[0]

    def timeFromSeconds(self, secs):
        '''
            This routine returns a datetime object for a given number of seconds

            Args:
                secs: number of seconds

            Returns:
                None

            Raises:
                None
        '''

        return datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=int(secs))

    def timeFromMinutes(self, mins):
        '''
            This routine returns a datetime object for a given number of minutes

            Args:
                mins: number of minutes

            Returns:
                None

            Raises:
                None
        '''
        return self.timeFromSeconds(mins * 60)

    def createCharts(self, df, filename):
        '''
            This routine creates multiple charts for future analysis.

            Args:
                df: Dataframe use for creating the charts
                filename: filename where to save the charts

            Returns:
                None

            Raises:
                None
        '''
        distance = 0.0
        elevation = df.at[0, 'elevation'] - df.at[0, 'slope'] * df.at[0, 'distance']
        durationMean = 0.0
        durationRegression = 0.0
        # chartDF is used for the charts
        chartDF = pd.DataFrame({
            'Distance': [distance],
            'Time by Mean': [durationMean],
            'Time by Mean (time)': ["{:02}:{:02}:{:02}" .format(self.timeFromMinutes(durationMean).hour, self.timeFromMinutes(durationMean).minute, self.timeFromMinutes(durationMean).second)],
            'Time by Regression': [durationRegression],
            'Time by Regression (time)': ["{:02}:{:02}:{:02}" .format(self.timeFromMinutes(durationRegression).hour, self.timeFromMinutes(durationRegression).minute, self.timeFromMinutes(durationRegression).second)],
            'Elevation': [round(elevation)]
        })

        for index, each_row in df.iterrows():
            distance += each_row['distance'] / 1000
            elevation = each_row['elevation']
            durationMean += each_row['Stop Factor'] * each_row['Time by Mean'] / 60
            durationRegression += each_row['Stop Factor'] * each_row['Time by Regression'] / 60
            dMean = self.timeFromMinutes(durationMean*60)
            dRegression = self.timeFromMinutes(durationRegression*60)
            df_row = pd.DataFrame([[round(elevation), distance, durationMean, durationRegression, "{:02}:{:02}:{:02}" .format(dMean.hour, dMean.minute, dMean.second), "{:02}:{:02}:{:02}" .format(dRegression.hour, dRegression.minute, dRegression.second)]], columns=['Elevation', 'Distance', 'Time by Mean', 'Time by Regression', 'Time by Mean (time)', 'Time by Regression (time)'])
            chartDF = pd.concat([chartDF, df_row], ignore_index=True)

        fig, axs = plt.subplots(3, 1, figsize=(8, 24))
        fig.suptitle("Covering of distance through time", fontsize=16)

        axs[0].plot(chartDF['Distance'], chartDF['Elevation'], 'tab:red')
        axs[0].set_title("Course Profile ", fontdict={'fontsize': 14, 'color': 'black'})
        axs[0].set_xlabel("Distance (kms)", fontdict={'fontsize': 12, 'color': 'black'})
        axs[0].set_ylabel("Elevation (mts)", fontdict={'fontsize': 12, 'color': 'black'})
        axs[0].set_xlim(0, max(chartDF['Distance']) * 1.02)
        axs[0].set_ylim(min(chartDF['Elevation']) * 0.98, max(chartDF['Elevation']) * 1.02)
        axs[0].grid()

        axs[1].plot(chartDF['Time by Mean'], chartDF['Distance'], 'tab:orange')
        axs[1].set_title("Estimated Time by Mean ", fontdict={'fontsize': 14, 'color': 'black'})
        axs[1].set_xlabel("Time (hours)", fontdict={'fontsize': 12, 'color': 'black'})
        axs[1].set_ylabel("Distance (kms)", fontdict={'fontsize': 12, 'color': 'black'})
        axs[1].set_xlim(0, max(chartDF['Time by Mean']) * 1.02)
        axs[1].set_ylim(0, max(chartDF['Distance']) * 1.02)
        axs[1].grid()

        axs[2].plot(chartDF['Time by Regression'], chartDF['Distance'], 'tab:green')
        axs[2].set_title("Estimated Time by Regression", fontdict={'fontsize': 14, 'color': 'black'})
        axs[2].set_xlabel("Time (hours)", fontdict={'fontsize': 12, 'color': 'black'})
        axs[2].set_ylabel("Distance (kms)", fontdict={'fontsize': 12, 'color': 'black'})
        axs[2].set_xlim(0, max(chartDF['Time by Regression']) * 1.02)
        axs[2].set_ylim(0, max(chartDF['Distance']) * 1.02)
        axs[2].grid()

        plt.savefig(filename)
        chartDF.drop(['Time by Mean', 'Time by Regression'], axis=1, inplace=True)
        print(chartDF)
        return chartDF

    def save(self, path):
        '''
            This routine saves to disk the regression models

            Args:
                path: filename of the file to save to

            Returns:
                None

            Raises:
                None
        '''

        for i in range(len(self.regression_model)):
            self.regression_model[i]['model'].save(path)