import os
import json
import geopy.distance
import geopandas

import pandas as pd
import numpy as np

from shapely.geometry import Point


"""

Class that handles reading JSON files from scooter companies as well as
finding trips, discritizing lat and lon, finding if points are in geojson, getting 
trajectory, and saving as csv. The underlying data structure is a pandas data frame
when a function is called that manipulates data such as to_trips() a new Dataframe is
returned. In order to chain operations you must make a new ScooterData object with the
dataframe. In future it might be better to return an instance of a ScooterData object
with such functions. I have not implemented this yet as I'm unsure if it's the best method.

"""


class ScooterData:

    df = pd.DataFrame()

    # When intialized a ScooterData object can either take an exisiting DataFrame
    # or it can take a dictionary that contains two values the first being 'item_labels'
    # which is a string with the label for each part of the file name. For example
    # if one of my file name is 'boston-bird-120312310-70' my item labels would reflect
    # what I want each column to be called. So if I want the first column to be called city
    # I would pass 'city-...' and if I wanted the next column to be company I would make my 
    # item label string 'city-company-...' and so on. The next item in the dictionary is
    # the path to the folder containing all the json files. There is also a sample perameter
    # which lets you pick a sample of data used. This is good for testing.
    def __init__(self, df=None, read_json_dict=None, sample=None):
        if read_json_dict is not None:
            item_labels = read_json_dict['item_labels']
            path = read_json_dict['path']
            self.df = self.get_files(path, self.get_file_names(path), item_labels)
            if sample is not None:
                self.df = self.df.sample(sample)
        elif df is not None:
            self.df = df
            if sample is not None:
                self.df = self.df.sample(sample) 


    def read_csv(self, path):
        self.df = pd.read_csv(path)

    # Gets all files in directory given by path
    def get_file_names(self, path):
        return os.listdir(path)

    # Helper function that gets all files in a directory
    def get_files(self, path, names, item_labels):
        all_data = []
        for name in names: # Runs through all file names
            if len(name.split('-')) < 4: # Checks that name has 4 or more '-'
                continue
            curr_path = os.path.join(path, name) # Gets full path with file name
            with open(curr_path) as f:
                data = json.load(f)
                # Skip data doesn't have the same structure so it has to be done a bit different
                if 'skip' in name: 
                    all_data += self.insert_city_provider_time(data['bikes'], name, item_labels)
                else:
                    all_data += self.insert_city_provider_time(data['data']['bikes'], name, item_labels)
        df = pd.DataFrame(all_data) # Creates DataFrame
        df = df.sort_values(by=['time'])
        df['time'] = pd.to_datetime(df['time'], unit='s') # Converts Unix time to Pandas datetime
        df = df.set_index('time') # Sets time as index
        df['time'] = df.index 
        df = df.tz_localize('UTC').tz_convert('US/Eastern') # Sets time zone
        df['hour'] = df.index.hour
        return df


    def insert_city_provider_time(self, list_of_dicts, file_name, item_labels):
        name_list = file_name[:-5].split('-') # Splits file_name into list of strings
        labels = item_labels.split('-')
        for x, item in enumerate(name_list):
            list_of_dicts = self.insert_info_list_dicts(list_of_dicts, labels[x], item)
        return list_of_dicts


    def insert_info_list_dicts(self, list_of_dicts, info_name, info):
        ret = []
        for dict in list_of_dicts:
            ret.append(self.insert_info_dict(dict, info_name, info))
        return ret


    def insert_info_dict(self, dict, info_name, info):
        dict[info_name] = info
        return dict


    def get_df(self):
        return self.df


    def set_df(self, df):
        self.df = df


    # Turns scooter location data over time into trips
    def to_trips(self, loc_change_thresh=.2):
        df = self.df.sort_index()
        bike_ids = set(df['bike_id'].values.tolist()) # Gets all unique bike_id
        trip_list = []
        for bike_id in bike_ids:
            df_temp = df[df['bike_id'] == bike_id] # makes df for only one scooter
            trip_points = set()
            for x in range(0, len(df_temp) - 1):
                p1 = [df_temp['lat'].iloc[x], df_temp['lon'].iloc[x]] # Gets earlier point
                p2 = [df_temp['lat'].iloc[x + 1], df_temp['lon'].iloc[x + 1]] # Gets next point
                dist = geopy.distance.distance(p1, p2).miles
                if dist > loc_change_thresh:
                    hour = df_temp['hour'].iloc[x]
                    trip_points.add((x, x + 1, dist, hour))
            for trip in trip_points:
                df_trip = df_temp.iloc[trip[0]:trip[1] + 1, :]
                trip_dict = {
                    'start_lat': df_trip.iloc[0]['lat'],
                    'start_lon': df_trip.iloc[0]['lon'],
                    'start_time': df_trip.iloc[0]['time'],
                    'end_time': df_trip.iloc[1]['time'],
                    'end_lat': df_trip.iloc[1]['lat'],
                    'end_lon': df_trip.iloc[1]['lon'],
                    'city': df_trip.iloc[1]['city'],
                    'bike_id': str(df_trip.iloc[1]['bike_id']),
                    'provider': str(df_trip.iloc[1]['provider']),
                    'trip_distance': trip[2],
                    'hour': trip[3]
                }
                trip_list.append(trip_dict)
        return pd.DataFrame.from_dict(trip_list)

    # Finds if points are in a geojson region or not
    # right now it just looks for geometry in zone_row which might not always work
    # and might need to be changed depending on the geojson file. column_val_dict is a 
    # dictionary that stores the new Dataframe column name as a key and the geojson property
    # name as the value for example if my geojson had a property 'AREA_TYPE' I would right
    # column_val_dict['col_name'] = 'AREA_TYPE'
    def to_geojson_discretize(self, geojson, column_val_dict):
        df = self.df.copy().reset_index()
        print(df)
        gdf = geopandas.GeoDataFrame(df, 
                geometry=[Point(float(x), float(y)) for x, y in zip(df.lon, df.lat)])
        for x, row in gdf.iterrows():
            point = row['geometry']
            for _, zone_row in geojson.iterrows():
                if zone_row['geometry'] and zone_row['geometry'].contains(point):
                    # print(zone_row)
                    for df_col, zone_col in column_val_dict.items():
                        df.loc[df.index[x], df_col] = zone_row[zone_col]
        return df


    def to_rounded_discretize(self, round_digits=3):
        df = self.df.copy()
        print(df.columns)
        print(round(df.lat[0], round_digits), df.lon)
        df['lat_disc'] = df['lat'].apply(lambda x: round(float(x), round_digits))
        df['lon_disc'] = df['lon'].apply(lambda x: round(float(x), round_digits))
        return df


    def get_trajectory(self):
        return self.df.apply(self._get_trajectory, axis=1)
    

    def _get_trajectory(self, row):
        p1 = np.array([row['start_lat'], row['start_lon']]).astype('float')
        p2 = np.array([row['end_lat'], row['end_lon']]).astype('float')
        row['vector_x'] = self.calc_vector(p1, p2)[0]
        row['vector_y'] = self.calc_vector(p1, p2)[1]
        return row


    def calc_vector(self, p1, p2):
        return p2 - p1

    # Saves as csv name must be specified you can also pick a column to 
    # sort by and whether it should be ascending or descending default is descending
    def to_csv(self, name, sort_col=None, ascending=False):
        df = self.df.copy()
        if sort_col is not None:
            df = df.sort_values(by=sort_col, ascending=ascending)
        df.to_csv(name)



# Example of how to turn json files into trip dataframe
def trip_test():
    test_dict = {
        'item_labels':'city-time-provider-batch',
        'path':'DC_6.14_'
    }

    sd = ScooterData(read_json_dict=test_dict, sample=10000)
    trips = ScooterData(sd.to_trips(.2))
    print(trips.get_df())


def geojson_disc_test():
    test_dict = {
        'item_labels':'city-time-provider-batch',
        'path':'DC_6.14_'
    }

    sd = ScooterData(read_json_dict=test_dict)

    test_dict = {
        'zone_label':'ZONING_LAB',
        'geojson_object_id':'OBJECTID'
    }

    zones = geopandas.read_file('Zoning_Regulations_of_2016.json')
    print(sd.to_geojson_discretize(zones, test_dict))


def rounded_disc_test():
    test_dict = {
        'item_labels':'city-time-provider-batch',
        'path':'DC_6.14_'
    }

    sd = ScooterData(read_json_dict=test_dict, sample=100)
    print(sd.to_rounded_discretize(3)[['lat', 'lon', 'lat_disc', 'lon_disc']])


def calc_vector_test():
    test_dict = {
        'item_labels':'city-time-provider-batch',
        'path':'DC_6.14_'
    }

    sd = ScooterData(read_json_dict=test_dict, sample=10000)
    trips = ScooterData(sd.to_trips(.2))
    print(trips.get_df())
    print(trips.get_trajectory())


def make_csv_test():
    test_dict = {
        'item_labels':'city-time-provider-batch',
        'path':'DC_6.14_'
    }

    sd = ScooterData(read_json_dict=test_dict, sample=10000)
    trips = ScooterData(sd.to_trips(.2))
    trips.to_csv('trip_test.csv', sort_col='start_time')


def read_csv_test():
    csv_path = 'trip_test.csv'

    sd = ScooterData()
    sd.read_csv(csv_path)
    print(sd.get_df())

read_csv_test()