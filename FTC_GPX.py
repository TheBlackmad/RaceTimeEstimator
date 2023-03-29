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

import json
import datetime
import matplotlib.pyplot as plt
from geopy import distance
import haversine
import gpxpy.gpx
import pandas as pd
import pytz
import requests
from dateutil import tz
from math import sqrt, floor


weight = 68.5
filename_corrections_source = "corrections-source.csv"
filename_corrections = "corrections.csv"
filename_pace_slope = "pace_slope.csv"

def get_weather(point):
    """
        Returns the weather conditions at this specific GPX point. Weather is specific for that hour in this point.
        The weather information is retrieve via the web service from www.weatherbit.io

        Parameters
        ----------
        point: point where to get weather on

        """

    global weather

    # initialize variables if no weather has been retrieved so far.
    if weather == None:
        # Get the weather data for that position (lat, lon) at that specific date and hour (mins and secs are omitted
        lat = point.latitude
        lon = point.longitude
        start_date = "%s-%s-%s" % (point.time.year, point.time.month, point.time.day)
        t = point.time + datetime.timedelta(days=1)
        end_date = "%s-%s-%s" % (t.year, t.month, t.day)

        url = url_weatherbit % (lat, lon, start_date, end_date, api_key_weatherbit)
        response = requests.get(url)
        weather = json.loads(response.text)
    #		print (weather)
    #		print(url)
    #		print('Response from WeatherBit API: ', weather)

    # The weather information is already retrieved
    data = weather['data']
    for entry in data:
        t = datetime.datetime.strptime(entry['timestamp_local'], '%Y-%m-%dT%H:%M:%S')

        # if same hour, return the weather conditions for that hour
        if point.time.year == t.year and point.time.month == t.month and point.time.day == t.day and point.time.hour == t.hour:
            return entry

    weather = None

    return None

def estimate_temp_altitude(ele, ele_base, temp_base, rh_base):
    """
        Returns the estimated temperature at the given elevation.
        This is an estimated temperature given the relative humidity rh and temperature at a given base elevation.
        if the relative humidity is high, then the temp is assumed to change with altitude as 3.3°C / 1000mts
		otherwise at a dry day, it changes as per 5.6°C / 1000mts
		Source: https://newsonthesnow.com/news/does-elevation-affect-temperature/

        Parameters
        ----------
        ele: elevation at which to estimate temperature in m
        ele_base: elevation base in m
        temp_base: temperature at the ele_base elevation in °C
        rh_base: relative humidity at ele_base elevation in %

        """

    if rh_base > 65:
        factor = -3.3
    else:
        factor = -5.6

    return (temp_base + factor * (ele - ele_base) / 1000.0)

class FTC_Route:
    '''
        This class represents a route or a track where we want to estimate the time for the race
        The GPX file can be parsed. The read methods can split the route in parts of ascend/descent according
        to a predefine gradient.
    '''

    def __init__(self, filename):
        '''
            This routine initialize the FTC_Route object with info from the GPX.

            Args:
                filename: GPX filename with the route

            Returns:
                self

            Raises:
                Exception: Error reading route file
        '''
        try:
            with open(filename, 'r') as gpx_file:
                self.gpx_route = gpxpy.parse(gpx_file)

        except:
            raise Exception ("Error reading route file")

    def read(self, stepDistance=80):
        '''
            This method reads the GPX with the route and splits it in sections of similar slopes

            Args:
                stepDistance: minimum distance of the sections in mts

            Returns:
                (elevation at start, list of section distances, list of section slopes)

            Raises:
                None
        '''

        distances = []
        slopes = []
        startElevation = self.gpx_route.tracks[0].segments[0].points[0].elevation

        # get the sections at stepDistance and optimize the list
        myList = self.__stepRoute(stepDistance=stepDistance)
        opt = self.__optimizer(myList)

        for item in opt:
            distances.append(item[0])
            slopes.append(item[1]/item[0])

        return startElevation, distances, slopes

    def __categorize(self, pdte):
        '''
            This method provides a classification of the slope, between [-5,5] depending on the grade

            Args:
                pdte: slope

            Returns:
                grade between [-5, 5]

            Raises:
                None
        '''
        if pdte < -40:
            return -5
        elif pdte < -25:
            return -4
        elif pdte < -15:
            return -3
        elif pdte < -8:
            return -2
        elif pdte < -1:
            return -1
        elif pdte < 1:
            return 0
        elif pdte < 8:
            return 1
        elif pdte < 15:
            return 2
        elif pdte < 25:
            return 3
        elif pdte < 40:
            return 4
        else:  # pdte < 100:
            return 5

    def __createTuple(self, d, e):
        '''
            This method creates a tuple based upon its inputs.

            Args:
                d: distance
                e: elevation difference

            Returns:
                (distance, elevation, slope, category)

            Raises:
                None
        '''
        pdte = e / d * 100  # pdte in %
        r = (d, e, pdte, self.__categorize(pdte))  # this split: how long and how steep
        return r

    # stepDistances
    def __stepRoute(self, stepDistance=100):
        '''
            This method creates a list of sections for the route with a stepDistance size for each section and includes
            the elevation gain for each section.

            Args:
                stepDistance: size in meters of the section

            Returns:
                list of sections of size stepDistance with the elevation gain for each

            Raises:
                None
        '''
        split_dist = 0.0
        split_ele = 0.0
        i = 0
        split_list = []

        # create from the route a list of sections of size stepDistance with elevation gain for each section
        for track in self.gpx_route.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if i == 0:
                        pre_point = point
                    if i != 0:
                        # get the elevation and distance difference between 2 points
                        this_geopoint = (point.latitude, point.longitude)
                        previous_geopoint = (pre_point.latitude, pre_point.longitude)
                        split_dist += sqrt((distance.geodesic(previous_geopoint, this_geopoint).meters) ** 2 + (point.elevation - pre_point.elevation) ** 2)
                        split_ele += point.elevation - pre_point.elevation

                        # if the two points are apart at least stepDistance, include in the list
                        if split_dist >= stepDistance:
                            split_list.append(self.__createTuple(split_dist, split_ele))
                            split_dist = 0.0
                            split_ele = 0.0
                    pre_point = point
                    i += 1
        return split_list

    # Optimizer
    def __optimizer(self, split_list):
        '''
            This method optimizes the lists of sections given as parameter. The optimization consists of contiguous
            sections of same category to be merge into one bigger section with cumulated elevation gain.

            Args:
                split_list: list of sections not optimized

            Returns:
                list of sections optimized.

            Raises:
                None
        '''
        i = 0
        sum_dist = 0.0  # adds distances for the same split category
        sum_ele = 0.0
        opt_split_list = []
        for sp in split_list:
            if i == 0:
                pre_sp = sp

            # for similar slope category, join into one single item.
            if sp[3] == pre_sp[3]:  # similar steep, so add together
                sum_dist += sp[0]
                sum_ele += sp[1]

            else:  # start a new split
                # start with a new split, so the previous is complete and must be included
                opt_split_list.append(self.__createTuple(sum_dist, sum_ele))
                sum_dist = sp[0]
                sum_ele = sp[1]

            pre_sp = sp
            i += 1

        # we need to include the last split
        opt_split_list.append(self.__createTuple(sum_dist, sum_ele))

        return opt_split_list

class FTC_GPX:
    '''
        This class represents a workout including GPX points, HR and Power data. For each GPX Point, HR and Power
        data are provided.
    '''

    def __init__(self, filename):
        '''
            This routine initialize the FTC_GPX object with info from the GPX. It created  dataframe with the plain
            data from the gpx and an optimised version of it. It also creates moving data from the gpx. Data of
            up/downhill and cumulative distance is also included.

            Args:
                filename: GPX filename with the workout

            Returns:
                self

            Raises:
                Exception: None
        '''
        # initialize all class attributes
        self.route_df = None                    # DataFrame from GPXPy
        self.opt_route_df = None             # DataFrame from GPXPy OPTIMIZED
        self.gpx_mv = None                      # gpx from GPXPy GET MOVING DATA
        self.opt_UphillDownhill = (0.0, 0.0)    # Uphill / Downhill cummulative
        self.cumulative_distance = 0.0          # Distance cummulative
        self.timeCreated = None                  # Time when GPX created

        # Local variables
        gpx = None                      # gpx from GPXPy
        route_info = []                 # List of GPX points
        i = 0

        # parse the GPX file workflow
        with open(filename, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        # get moving data and GPX creation time
        self.gpx_mv = gpx.get_moving_data(raw=True, speed_extreemes_percentiles=0.05,
                                               ignore_nonstandard_distances=False)
        self.timeCreated = gpx.time

        # get all data for each point of the workout.
        for track in gpx.tracks:
            for segment in track.segments:
                previous_point = segment.points[0]
                for point in segment.points[1:]:
                    # calculate the weather conditions for that point
                    #				w = get_weather(point)

                    # estimates the temperature at the given elevation
                    #				temp_elev = estimate_temp_altitude (point.elevation, segment.points[0].elevation, w['temp'], w['rh'])
                    temp_elev = estimate_temp_altitude(point.elevation, segment.points[0].elevation, 0, 0)

                    this_geopoint = (point.latitude, point.longitude)
                    previous_geopoint = (previous_point.latitude, previous_point.longitude)
                    self.cumulative_distance += distance.geodesic(previous_geopoint, this_geopoint).meters

                    # calculate the slope on that point
                    if i == 0 or point.distance_3d(previous_point) == 0:
                        slope = 0.0
                    else:
                        slope = 100 * (point.elevation - previous_point.elevation) / \
                                point.distance_3d(previous_point)

                    # calculate the speed on that point
                    if segment.get_speed(segment.points.index(point)) is not None:
                        speed = segment.get_speed(segment.points.index(point)) * 3.6
                    else:
                        speed = 0.0

                    route_info.append({
                        'time': point.time,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'cumulative_distance': self.cumulative_distance,
                        'speed': speed,
                        'slope': slope,
                        'power': self.getPower(point),
                        'hr': self.getHR(point),
                        'cad': self.getCad(point),
                        #					'temp_base': w['temp'],
                        'temp': temp_elev,
                        # Here it is the potential use of giving weather conditions to each point, if linked to
                        # a weather service.
                        #					'rh': w['rh'],
                        #					'wind_spd': w['wind_spd'],
                        #					'wind_dir': w['wind_dir'],
                        #					'pres': w['pres'],
                        #					'slp': w['slp'],
                        #					'app_temp': w['app_temp'],
                        #					'dewpt': w['dewpt'],
                        #					'clouds': w['clouds'],
                        #					'pod': w['pod'],
                        #					'weather_desc': w['weather']['description'],
                        #					'vis': w['vis'],
                        #					'precip': w['precip'],
                        #					'snow': w['snow'],
                        #					'uv': w['uv']
                    })

        # create the DataFrame from the GPX and OPTIMIZED
        self.route_df = pd.DataFrame(route_info)
        self.opt_route_df = pd.DataFrame(self.optimizedGPX(gpx))

    def raw_to_csv(self, filename):
        '''
            This routine save to a csv file the data from the workout.

            Args:
                filename: filename where to store csv data

            Returns:
                None

            Raises:
                Exception: None
        '''
        # Create the dataframe and export to CSV
        print('Exporting to CSV . . .')
        self.route_df.to_csv(filename)
        print('Exporting to CSV . . . DONE')

    def print_Info_Raw(self):
        '''
            This routine displays basic information about the raw workout.

            Args:
                None

            Returns:
                None

            Raises:
                Exception: None
        '''
        print('Get uphills/downhills: ', self.gpx.get_uphill_downhill())
        print('Get duration: ', self.gpx.get_duration() / 60)
        print('Get moving distance: ', self.gpx_mv.moving_distance / 1000)
        print('Get moving time: ', self.gpx_mv.moving_time / 60)
        print('Get max speed: ', self.gpx_mv.max_speed)

        print('Number of data Points: ', self.gpx.get_track_points_no())
        print('Get extreme elevation values: ', self.gpx.get_elevation_extremes())

        print('Track moving data: ', self.gpx.tracks[0].get_moving_data())

        print('Segment moving data: ', self.gpx.tracks[0].segments[0].get_moving_data())
        print('Segment length 2D data: ', self.gpx.tracks[0].segments[0].length_2d())
        print('Segment length 3D data: ', self.gpx.tracks[0].segments[0].length_3d())
        print('Completing route info with further data . . .')

    def plot(self):
        '''
            This routine plots speed averages for the workout.

            Args:
                None

            Returns:
                None

            Raises:
                Exception: None
        '''
        # Plot playtime
        self.route_df["speed_raw"] = self.route_df["speed"].rolling(window=1, min_periods=1, center=True).mean()
        self.route_df["speed_ma_10"] = self.route_df["speed"].rolling(window=10, min_periods=1, center=True).mean()
        self.route_df["speed_ma_30"] = self.route_df["speed"].rolling(window=30, min_periods=1, center=True).mean()
        self.route_df["speed_ma_60"] = self.route_df["speed"].rolling(window=60, min_periods=1, center=True).mean()
        self.route_df["speed_ma_120"] = self.route_df["speed"].rolling(window=120, min_periods=1, center=True).mean()
        self.route_df["speed_ma_300"] = self.route_df["speed"].rolling(window=300, min_periods=1, center=True).mean()
        # route_df['speed_ma'].plot(figsize=(15,0))
        self.route_df[0:30900][['speed_raw', 'speed_ma_30', 'speed_ma_60']].plot(figsize=(15, 8))
        self.route_df[0:30900][['speed_ma_30']].plot(figsize=(15, 8))

        # Plot diagram for temp and speed
        plt.figure(figsize=(14, 8))
        # plt.scatter(route_df['number'], route_df['speed'])
        plt.plot(self.route_df['cumulative_distance'], self.route_df['speed'])
        plt.title('Relationship distance and speed', size=20)
        plt.show()

    def optimizedGPX(self, gpx):
        '''
            This routine generates further information about the workout provided as argument and returns it in a new
            gpx object. This includes in a new list information about the Grade Adjusted Pace and Speed (GAP / GAS),
            efficiency, power, calories consumed, etc. It intends to provide a way to estimate own metrics for the
            given data.

            Also, this routine considers relevant the distances every 10 seconds and not less, as the errors can be
            big and accumulated. Therefore, 1 in 10 samples of the gpx are ignored.The errors in the height can be
            bigger, and in the overall cumulative height is considered every 24 seconds, instead.

            It provides all this information in a new gpx object at the expense of memory consumption, but
            it allows to keep original data be kept as 'untouched' data.

            Args:
                gpx: GPX workout

            Returns:
                optimised / extended version of gpx

            Raises:
                Exception: None
        '''
        opt_route = []
        my_gpx = None
        dist_vin_no_alt = [0]
        dist_hav_no_alt = [0]
        dist_vin = [0]
        dist_hav = [0]
        dist_dif_vin_2d = [0]
        dist_dif_hav_2d = [0]
        alt_dif = [0]
        dist_dif_per_sec = [0]
        time_dif = []

        # create a new gpx object that will be used for the optimzed output.
        my_gpx = gpxpy.gpx.GPX()

        # Create first track in our GPX:
        my_gpx_track = gpxpy.gpx.GPXTrack()
        my_gpx.tracks.append(my_gpx_track)

        # Create first segment in our GPX track:
        my_gpx_segment = gpxpy.gpx.GPXTrackSegment()
        my_gpx_track.segments.append(my_gpx_segment)

        # Create points:
        count = 0
        for track in gpx.tracks:
            for segment in track.segments:
                # Here the content of the GPX is in UTC time format
                # else take UTC as default
                if segment.points[0].time.tzname() == "Z":
                    last_time = datetime.datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=pytz.UTC)
                else:
                    last_time = datetime.datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=tz.gettz('Europe / Berlin'))

                last_ele = segment.points[0].elevation

                # take 1 point out of 10. This will reduce the cumulative errors.
                for point in segment.points:
                    td = point.time - last_time

                    # to increase the precision of the measurements,
                    # it takes the positions every 10 seconds.
                    # to calculate distances
                    if td.seconds >= 10:
                        last_time = point.time
                        # include point in the route
                        my_gpx_segment.points.append(point)

                # this section is to get the tuple (cumulative uphill, cumulative downhill)
                last_time = datetime.datetime(2000, 1, 1, 0, 0, 0, 1, tzinfo=pytz.UTC)
                for point in segment.points:
                    td = point.time - last_time

                    # based on experience, it took for the elevation only
                    # samples every 24 seconds. it seems to be a good approach to the cumulative elevation
                    # the reason is that ascending is not that quick as moving horizontal and it needs higher times
                    # for sampling to balance with errors.
                    if td.seconds >= 24:
                        last_time = point.time
                        count = count + 1
                        # update values uphill and downhill
                        delta_ele = point.elevation - last_ele
                        if delta_ele >= 0:
                            self.opt_UphillDownhill = (self.opt_UphillDownhill[0] + delta_ele, self.opt_UphillDownhill[1])
                        else:
                            self.opt_UphillDownhill = (self.opt_UphillDownhill[0], self.opt_UphillDownhill[1] - delta_ele)

                        last_ele = point.elevation

                my_gpx_segment.points.append(segment.points[segment.get_points_no() - 1])

        # this section is to build the list with the output optimized list of points.
        # though going through the gpx points in several occassions, it is clear it can be done in a very much
        # quicker manner. It is intended to optimze code in a future version, but this is intended to be done so,
        # for the shake of clarity.
        for track in my_gpx.tracks:
            for segment in track.segments:
                start_point = segment.points[0]
                previous_point = segment.points[0]
                for point in segment.points[1:]:

                    startPoint = (previous_point.latitude, previous_point.longitude)
                    stopPoint = (point.latitude, point.longitude)

                    # Calculate 2D distances using Vincenty (geodesic) and Haversine
                    distance_vin_2d = distance.geodesic(startPoint, stopPoint).m
                    dist_dif_vin_2d.append(distance_vin_2d)
                    dist_vin_no_alt.append(dist_vin_no_alt[-1] + distance_vin_2d)
                    distance_hav_2d = haversine.haversine(startPoint, stopPoint) * 1000
                    dist_dif_hav_2d.append(distance_hav_2d)
                    dist_hav_no_alt.append(dist_hav_no_alt[-1] + distance_hav_2d)

                    # calculate elevations for each Vincenty and Haversine
                    alt_d = point.elevation - previous_point.elevation
                    alt_dif.append(alt_d)

                    # calculate 3D distances using Vincenty (geodesic) and Haversine
                    distance_vin_3d = sqrt(distance_vin_2d ** 2 + alt_d ** 2)
                    dist_vin.append(dist_vin[-1] + distance_vin_3d)
                    distance_hav_3d = sqrt(distance_hav_2d ** 2 + alt_d ** 2)
                    dist_hav.append(dist_hav[-1] + distance_hav_3d)

                    # calculate time deltas between points for later speed calculations
                    time_delta = (point.time - previous_point.time).total_seconds()
                    time_dif.append(time_delta)

                    # calculate speeds at every point
                    if time_delta > 0.0:
                        distdifpersec = distance_vin_2d / time_delta
                    else:
                        distdifpersec = 0.0
                    dist_dif_per_sec.append(distdifpersec)

                    # calculate slope, speed and pace
                    if distance_vin_2d != 0:
                        slope = alt_d / distance_vin_2d
                        speed = (distdifpersec) * 3.6
                        pace = 60.0 / speed
                    else:
                        slope = 0.0
                        speed = 0.0
                        pace = 0.0

                    # calculate GAP factor, and in adjusted speed and pace
                    gap_factor = 0.0021 * (slope * 100) * (slope * 100) + 0.034 * slope * 100 + 1
                    if pace != 0.0:
                        gap = pace / gap_factor
                        gas = 60.0 / gap
                    else:
                        gap = 0.0
                        gas = 0.0

                    # Get efficiency metrics
                    if self.getHR(point) == 0:
                        efficiency = 0.0
                        efficiencyPower = 0.0
                        efficiencyCadence = 0.0
                    else:
                        efficiency = (1000 * gas / 60.0) / self.getHR(point)
                        efficiencyPower = self.getPower(point) / self.getHR(point)
                        efficiencyCadence = self.getCad(point) / self.getHR(point)

                    # create list with information for each point
                    opt_route.append({
                        'time': point.time,
                        'timeabs': (point.time - start_point.time).total_seconds(),
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'dist_vin_no_alt': dist_vin_no_alt[-1],
                        'dist_hav_no_alt': dist_hav_no_alt[-1],
                        'alt_dif': alt_d,
                        'dist_vin': dist_vin[-1],
                        'dist_hav': dist_hav[-1],
                        'time_dif': time_delta,
                        'dist_dif_per_sec': distdifpersec,
                        'speed': speed,
                        'slope': slope,
                        'power': self.getPower(point),
                        'hr': self.getHR(point),
                        'cad': self.getCad(point),
                        'calories': self.getPower(point) / (360 * 0.3265),
                        'GAP_factor': gap_factor,
                        'pace': pace,
                        'GAP': pace / gap_factor,
                        'GAS': gas,
                        'efficiency': efficiency,
                        'calories_new': 0.865 * gas * weight * time_delta / 3600,
                        'efficiencyPower': efficiencyPower,
                        'efficiencyCadence': efficiencyCadence
                    })

                    previous_point = point

        return opt_route

    def print_Info_Optimized(self):
        '''
            This routine displays basic information about the optimized workout.

            Args:
                None

            Returns:
                None

            Raises:
                Exception: None
        '''
        print('New Segment moving data: ', self.my_gpx.tracks[0].segments[0].get_moving_data())
        print('New Segment length 2D data: ', self.my_gpx.tracks[0].segments[0].length_2d())
        print('New Segment length 3D data: ', self.my_gpx.tracks[0].segments[0].length_3d())

        print('Diff Segment length 2D data: ',
              self.my_gpx.tracks[0].segments[0].length_2d() - self.my_gpx.tracks[0].segments[0].length_2d())
        print('Diff Segment length 3D data: ',
              self.my_gpx.tracks[0].segments[0].length_3d() - self.my_gpx.tracks[0].segments[0].length_3d())

        print('New umber of data Points: ', self.my_gpx.get_track_points_no())
        print('New Get extreme elevation values: ', self.my_gpx.get_elevation_extremes())
        print('New Get uphills/downhills: ', self.my_gpx.get_uphill_downhill())
        print('New Get duration: ', self.my_gpx.get_duration() / 60)

        print('my uphill / downhill elevation: ', self.opt_UphillDownhill)

    def opt_to_csv(self, filename):
        '''
            This routine save to a csv file the data from the optimized workout.

            Args:
                filename: filename where to store csv data

            Returns:
                None

            Raises:
                Exception: None
        '''
        # Create the dataframe and export to CSV
        print('Exporting to CSV . . .')
        self.opt_route_df.to_csv(filename)
        print('Exporting to CSV . . . DONE')

    def getVincenty2D(self, route):
        '''
            This routine returns Vincenty 2D distance of the workout.

            Args:
                None

            Returns:
                Vincenty 2D distance

            Raises:
                Exception: None
        '''
        return route[-1]['dist_vin_no_alt']

    def getVincenty3D(self, route):
        '''
            This routine returns Vincenty 3D distance of the workout.

            Args:
                None

            Returns:
                Vincenty 3D distance

            Raises:
                Exception: None
        '''
        return route[-1]['dist_vin']

    def getHaversine2D(self, route):
        '''
            This routine returns Haversine 2D distance of the workout.

            Args:
                None

            Returns:
                workout Haversine 2D distance

            Raises:
                Exception: None
        '''
        return route[-1]['dist_hav_no_alt']

    def getHaversine3D(self, route):
        '''
            This routine returns Haversine 3D distance of the workout.

            Args:
                None

            Returns:
                Haversine 3D distance

            Raises:
                Exception: None
        '''
        return route[-1]['dist_hav']

    def getElapsedTime(self):
        '''
            This routine returns the elapsed time of the workout.

            Args:
                None

            Returns:
                elapse time for the workout

            Raises:
                Exception: None
        '''
        return datetime.timedelta(seconds=int( max(self.opt_route_df['timeabs']) ))

    def getAvgSpeed(self):
        '''
            Returns the average speed for the workout

            Args:
                None

            Returns:
               timedelta

            Raises:
               Exception: None
        '''
        return self.opt_route_df['speed'].mean()

    def getAvgPace(self):
        '''
            Returns the average pace for the workout

           Args:
               None

           Returns:
               timedelta

           Raises:
               Exception: None
        '''
        t = 1 / (self.getAvgSpeed() / 60)
        return datetime.timedelta(minutes=floor(t), seconds=(t - floor(t)) * 60)

    def getMovingAvgSpeed(self, quantileLow=0.05, quantileHigh=1.0, stop_speed=3.0):
        '''
            Returns the moving average speed for the workout

           Args:
                quantileLow: lower quantile
                quantileHigh: upper quantile
                stop_speed: speed below which is considered no moving

            Returns:
                average speed in km/h

            Raises:
                Exception: None
        '''
        q_low = max(self.opt_route_df['speed'].quantile(quantileLow),
                    stop_speed)  # STRAVA considers no movement when speed is <= 3.0 km/h
        q_hi = self.opt_route_df['speed'].quantile(quantileHigh)
        df_with_timeout = self.opt_route_df[(self.opt_route_df['speed'] < q_hi) & (self.opt_route_df['speed'] > q_low)]
        #		print('Valor Quantile: ', q_hi, ' / ', q_low)
        #		print(df_with_timeout['speed'].count())
        #		print(df_with_timeout['speed'].max(), ' / ', df_with_timeout['speed'].min())

        avg_km_h = (sum((df_with_timeout['speed'] * df_with_timeout['time_dif'])) / sum(df_with_timeout['time_dif']))

        return avg_km_h

    def getMovingAvgPace(self, quantileLow=0.05, quantileHigh=1.0, stop_speed=3.0):
        '''
            Returns the moving average pace for the workout

           Args:
                quantileLow: lower quantile
                quantileHigh: upper quantile
                stop_speed: speed below which is considered no moving

            Returns:
                timedelta

            Raises:
                Exception: None
        '''
        avg_km_h = self.getMovingAvgSpeed(quantileLow, quantileHigh, stop_speed)
        return datetime.timedelta(minutes=floor(60 / avg_km_h),
                                  seconds=round(((60 / avg_km_h - floor(60 / avg_km_h)) * 60)))

    def getDataFrame(self):
        '''
            Returns the the optimized data frame for this workout

           Args:
                None

            Returns:
                DataFrame with the optimized route

            Raises:
                Exception: None
        '''
        return self.opt_route_df

    def getPower(self, point):
        '''
            Returns the power exercise at the GPX point. Power is provided by the GPX equipment.
            If not exist 0 is returned

            Args:
                point: GPX point

            Returns:
                exercise power for that point

            Raises:
                Exception: None
        '''
        for extension in point.extensions:
            if extension.tag == 'power':
                return int(extension.text)

        return 0

    def getHR(self, point):
        '''
			Returns the heart rate exercise at the GPX point. Heart rate is provided by the GPX equipment.
			If not exist 0 is returned

           Args:
                point: GPX point

            Returns:
                exercise HR for that point

            Raises:
                Exception: None
        '''
        for extension in point.extensions:
            if extension.tag != 'power':
                for exten in extension:
                    if exten.tag == '{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr':
                        return int(exten.text)

        return 0

    def getCad(self, point):
        '''
			Returns the cadence exercise at the GPX point. Cadence is provided by the GPX equipment.
			If not exist 0 is returned

           Args:
                point: GPX point

            Returns:
                exercise Cadence for that point

            Raises:
                Exception: None
        '''
        for extension in point.extensions:
            if extension.tag != 'power':
                for exten in extension:
                    if exten.tag == '{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}cad':
                        return int(exten.text)

        return 0

    def getUphillDownhill(self):
        '''
            Returns a tuple with the data included as (uphill, downhill). Cumulative values returned.

           Args:
                None

            Returns:
                tuple (uphill, downhill)

            Raises:
                Exception: None
        '''
        return ( round(self.opt_UphillDownhill[0], 0), round(self.opt_UphillDownhill[1], 0) )

    def getDate(self):
        '''
            Returns the time when this workout was created

           Args:
                None

            Returns:
                timeCreated

            Raises:
                Exception: None
        '''
        return self.timeCreated
