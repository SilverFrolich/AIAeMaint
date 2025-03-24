#%%
import time  
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
# Read the data file into a dataframe
df = pd.read_csv(r'IAIeMaint\Assignment3\train_delays-1.csv', index_col=0)  # Use read_csv() for CSV files
print(df)
print(df.shape)
#%%
#Clean data 
missing_values = df.isnull().sum() 
missing_values = missing_values[missing_values > 0] 
print("Columns with missing values:")
print(missing_values)
print("\nPercentage of missing values per column:")
print((missing_values / len(df)) * 100)
#%%
# Identify outliers in registered delay
median_delay = np.median(df['registered delay'])
mad_delay = np.median(np.abs(df['registered delay'] - median_delay))

df['modified_z_score'] = 0.6745 * (df['registered delay'] - median_delay) / mad_delay

threshold = 3.5
outliers = df[np.abs(df['modified_z_score']) > threshold]

num_outliers = outliers.shape[0]
print(f'Number of outliers removed: {num_outliers}')
df = df[np.abs(df['modified_z_score']) <= threshold]

#Remove reason code 1 since it is always the same
print(f'Remove Reason code - level 1')
df = df.drop(columns=["Reason code - level 1"], errors='ignore')

#When reason code 3 is empty, add "Not known"
df["Reason code Level 3"] = df["Reason code Level 3"].fillna("Inte angivet")
df.loc[df["Reason code Level 3"] == "", "Reason code Level 3"] = "Inte angivet"
df["Reason code Level 3"] = df["Reason code Level 3"].replace("-", "Inte angivet")
print(df["Reason code Level 3"].value_counts()) 

#%%
#Feature Engineering
#Determine location
geolocator = Nominatim(user_agent="train_delay_visualizer")
def get_coordinates(place_name):
    try:
        location = geolocator.geocode(place_name)
        if location:
            print(f"Location for {place_name}: lat {location.latitude} long {location.longitude}")
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding {place_name}: {e}")
        return None, None

unique_places = df['Place'].unique()

place_coords = {}
for place in unique_places:
    lat, lon = get_coordinates(place)
    place_coords[place] = (lat, lon)
    time.sleep(1)

df['latitude'] = df['Place'].map(lambda x: place_coords[x][0])
df['longitude'] = df['Place'].map(lambda x: place_coords[x][1])
#%%
#Fill Route with information
def fill_route_by_place(df):
    for index, row in df[df['Route'].isnull()].iterrows():
        similar_places = df[(df['Place'] == row['Place']) & (df['Route'].notnull())]
        
        if not similar_places.empty:
            most_common_route = similar_places['Route'].mode()[0]
            df.at[index, 'Route'] = most_common_route
    return df

def fill_route_by_location(df):
    for index, row in df[df['Route'].isnull()].iterrows():
        min_distance = float('inf')
        best_route = None
        
        for _, known_row in df[df['Route'].notnull()].iterrows():
            distance = geodesic((row['latitude'], row['longitude']), (known_row['latitude'], known_row['longitude'])).km
            
            if distance < min_distance:
                min_distance = distance
                best_route = known_row['Route']
        
        if best_route:
            df.at[index, 'Route'] = best_route
    return df

df = fill_route_by_place(df) 
df = fill_route_by_location(df) 

print("Null routes: ")
print(df['Route'].isnull().sum())  

#%%
#Feature engineering  
#Add day of the week
df["Train ID"] = df["Train ID"].astype(str)

df["date"] = df["Train ID"].str[:10]
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

df["day_of_week"] = df["date"].dt.day_name()
df['month'] = df['date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

print(df[["Train ID", "day_of_week", "season", "month"]].head())

#%%
print(df)
print(df.shape)
#%%
#Different plots
#Delay by train number
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Tågnr', y='registered delay', palette='viridis')
plt.title('Delay by Train Number')
plt.xlabel('Train Number')
plt.ylabel('Registered Delay ')
plt.show()
#%%
# Delay by route
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Route', y='registered delay', palette='viridis')
plt.title('Delay by Route')
plt.xlabel('Route')
plt.ylabel('Registered Delay')
plt.show()
#%%
# Registered Delay 
plt.figure(figsize=(10, 6))
sns.histplot(df['registered delay'], bins=50, kde=True)
plt.title('Distribution of Train Delays')
plt.xlabel('Delay Time (minutes)')
plt.ylabel('Frequency')
plt.show()
#%%
#Delay by day of week
day_count = df["day_of_week"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=day_count.index, y=day_count.values, palette="Set2")
plt.title("Delays by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Delays")
plt.xticks(rotation=45)
plt.show()
#%%
# Delay by month
month_count = df['month'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=month_count.index, y=month_count.values, palette="coolwarm")

plt.title("Delays by Month")
plt.xlabel("Month")
plt.ylabel("Number of Delays")
plt.xticks(ticks=range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.show()
#%%
# Delay by season
season_count = df['season'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=season_count.index, y=season_count.values, palette="viridis")

plt.title("Delays by Season")
plt.xlabel("Season")
plt.ylabel("Number of Delays")
plt.show()

#%%
#The place of the delay
geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]

gdf = gpd.GeoDataFrame(df, geometry=geometry)
gdf.set_crs("EPSG:4326", inplace=True)

fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, marker='o', color='red', markersize=5)

plt.title("Spatial Distribution of Train Delays")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
#%%
#Operator pie chart
operator_count = df['Operator'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(operator_count, labels=operator_count.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(operator_count)))

plt.title("Distribution of Train Operators")
plt.axis('equal')
plt.show()

#%% Preprocessing
# Define features and target
features = ['longitude', 'latitude', 'day_of_week', 'season', 'month', 'Reason code Level 3', 'Operator']
target = 'registered delay'

X = df[features]
y = df[target]

categorical_features = ['day_of_week', 'season', 'Reason code Level 3', 'Operator']
numeric_features = ['longitude', 'latitude', 'month']

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),  
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#%%
# Train the model
model.fit(X_train, y_train)

#%%
# Predict using the model
y_pred = model.predict(X_test)

#%%
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# %%
