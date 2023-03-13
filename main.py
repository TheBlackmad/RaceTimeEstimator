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

import os
import time

from datetime import datetime, timezone
import pandas as pd
from matplotlib import pyplot as plt

import FTC_GPX as g

path_for_data = "/home/afuerno/Dropbox/test-DB/"
#path_for_data = "../../PycharmProjects/pythonProject/allgpx/more/"
filename_params = "params.cfg"
filename_route = "route.gpx.info"
filename_chart = "chart.jpg"
filename_efficiency = "efficiency.jpg"

# retrieve the list of GPX files
currentList = set( filter( lambda item: item.endswith(".gpx") or item.endswith("*.GPX"), set( os.listdir(path_for_data) ) ) )
fileAdded = True
fileDelete = False
message = ""

# see if there is a config file, otherwise all files are going to be inspected.
d = datetime.utcnow()
model = g.RunningModel(path_for_data)
d1 = datetime.utcnow()
td = d1 - d
print("Time for Reading all data for the model: ", td)

newList = currentList
while 1:

    if fileAdded:
        # Reading the route file
        print("Reading the route ", flush=True)
        elevation, distances, slopes = g.FTC_Route(path_for_data + "/" + filename_route).read2()
        if distances is None:
            print("Error reading route.info file")
            exit(0)

        # Add the new file
        filenames = (currentList ^ newList)       # This is the new file
        for filename in list(filenames):
            d = datetime.utcnow()
            model.addNewWorkout(path_for_data+"/"+filename)
            d1 = datetime.utcnow()
            td = d1 - d
            print("Time for reading gpx: ", td)
        model.estimatePaceInRace(raceDistance=sum(distances), periodInDays=600, selection=1.0)
        d = model.timeFromMinutes(model.refGap)
        print("It has been predicted that for {:.2f}".format(model.refDistance / 1000), " Kms, the GAP would be {:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second), flush=True)
        message = f"It has been predicted that for {round(model.refDistance/1000, 2)} Kms, the GAP would be " + "{:02}:{:02}:{:02} mins/km" .format(d.hour, d.minute, d.second) + "\n" + "\n"

        # Training the model
        print ("Training the model ", flush=True)
        d = datetime.utcnow()
        model.DanielPaceModelFit(periodInDays=365)
        d1 = datetime.utcnow()
        print (f"Training complete. Time needed for training the model: {d1-d}", flush=True)

        # Calculate race time based on Riegel
        d = model.timeFromMinutes( model.Riegel(sum(distances), model.refDistance, model.refGap*model.refDistance/1000) )
        print("ESTIMATED TIME FOR", round(sum(distances)/1000, 2), "Kms (Riegel's Model): ", "{:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second))
        message += f"ESTIMATED TIME FOR {round(sum(distances)/1000,2)} Kms (Riegel's Model): " + "{:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second) + "\n"

        # Calculate race time based on Daniel
        d = model.timeFromMinutes( model.Daniel(distances, slopes, secsPerGradeplusPerKm=21.3, secsPerGrademinusPerKm=-2.13) )
        print("ESTIMATED TIME FOR", round(sum(distances)/1000, 2), "Kms (Daniel's Model): {:02}:{:02}:{:02} with Correctors 21.3/-2.13" .format(d.hour, d.minute, d.second))
        message += f"ESTIMATED TIME FOR {round(sum(distances)/1000,2)} Kms (Daniel's Model): " + "{:02}:{:02}:{:02} with Correctors 21.3/-2.13" .format(d.hour, d.minute, d.second) + "\n"

        # Calculate race time based on my own Model, both using Mean Pace and Regression
        corr, myTimeByMean, myTimeByRegression = model.DanielByModel(distances, slopes, elevation, periodInDays=365)
        d = model.timeFromMinutes(myTimeByMean)
        print("ESTIMATED TIME FOR", round(sum(distances)/1000, 2), "Kms (Runner's model based upon Mean Pace): ", "{:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second))
        message += f"ESTIMATED TIME FOR {round(sum(distances)/1000,2)} Kms (Runner's model based upon Mean Pace): " + "{:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second) + "\n"
        d = model.timeFromMinutes(myTimeByRegression)
        print("ESTIMATED TIME FOR", round(sum(distances)/1000, 2), "Kms (Runner's model based upon Regression): ", "{:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second))
        message += f"ESTIMATED TIME FOR {round(sum(distances)/1000, 2)} Kms (Runner's model based upon Regression): " + "{:02}:{:02}:{:02}" .format(d.hour, d.minute, d.second) + "\n" + "\n"

        # Display details for the track
        df = pd.DataFrame(corr)
#        print(f"Display details of the subtracks:\n {df.to_string()}")
#        message += f"Display details of the subtracks:\n {df.to_string()}" + "\n"

        print(f"Total distance of the target race: {round(df['distance'].sum(),2)}")
        message += f"Total distance of the target race: {round(df['distance'].sum(),2)}" + "\n\n"

        # show fitness
        lst = []
        for workout in model.workouts:
            daysAgo = (workout['date'] - datetime.now(timezone.utc)).days
            ef = workout['EfficiencyFactor']
            lst.append({
                'days ago': daysAgo,
                'months ago': daysAgo/(365/12),
                'date': workout['date'],
                'EfficiencyFactor': ef,
            })
        WDF = pd.DataFrame(lst).sort_values(by=['days ago'])
        WDF['EF-ema'] = WDF['EfficiencyFactor'].ewm(span=10).mean()
        WDF.plot(x='months ago', y=['EfficiencyFactor', 'EF-ema'], kind='line', figsize=(15,10), grid=True)
        plt.title('Efficiency over Time')
        plt.savefig(path_for_data+"/"+filename_efficiency)

        fileAdded = False
        fileDelete = False

    time.sleep(5)
    newList = set( filter( lambda item: item.endswith(".gpx") or item.endswith("*.GPX"), set( os.listdir(path_for_data) ) ) )

    # what if new files are identified
    if len(currentList - newList) != 0:
        print(f"File deleted: ", currentList - newList)
        if os.path.isfile(path_for_data+filename_params):
            os.remove(path_for_data+filename_params)
        fileDelete = True
    elif len(currentList ^ newList) != 0:
        print(f"File added: ", currentList ^ newList)
        if os.path.isfile(path_for_data+filename_params):
            os.remove(path_for_data+filename_params)
        fileAdded = True

    currentList = newList
