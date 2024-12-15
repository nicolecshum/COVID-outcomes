# %% [markdown]
# # Stringency Index (Our World in Data)
# 

# %%
import requests
import pandas as pd
import json

# URL for the Stringency Index JSON file
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/internal/megafile--stringency.json"

# Fetch the JSON data
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Convert JSON response to a Python dictionary
    raw_json = response.json()  # Convert the response to a dictionary
    print("Data successfully retrieved.")
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")


# %%
# Print top-level keys in the dictionary
raw_json.keys()


# %%
# Covert to pandas data frame
df = pd.DataFrame(raw_json)
df.head()

# %%
# Retain only necessary columns for simplicity
df = df[['location', 'date', 'stringency_index']]
    
 # Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])
    
# Extract year from the date column
df['year'] = df['date'].dt.year

# Filter data for the years 2020, 2021, 2022
df = df[df['year'].isin([2020, 2021, 2022])]

# Group data by location and year, aggregating the stringency index
aggregated_df = df.groupby(['location', 'year'], as_index=False)['stringency_index'].mean()

# Display the aggregated DataFrame
aggregated_df.head()



# %% [markdown]
# # World Bank API

# %%
import requests

# Initial attempt to query World Bank API
url = "https://search.worldbank.org/api/v3/wds"
params = {
    "format": "json",  # Request JSON format
    "qterm": "health",  # Search keyword
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    print("Data retrieved successfully.")
except requests.exceptions.RequestException as e:
    print(f"Initial request failed: {e}")
    




# %% [markdown]
# The query returned a generic dataset with irrelevant fields and missing key socioeconomic indicators.

# %%
# Second attempt to query World Bank API


# API URL for USA's GDP data
url = "https://api.worldbank.org/v2/country/usa/indicator/NY.GDP.MKTP.CD"
params = {
    "format": "json",   # Response format
    "date": "2020:2022" # Time range
}

# Request the data
try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Parse the response into a DataFrame
    records = [
        {"Year": entry["date"], "GDP (current US$)": entry["value"]}
        for entry in data[1]
    ]
    gdp_df = pd.DataFrame(records)
    print(gdp_df)
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")


# %%
# Third Attempt: using 'all' 

# Define the API endpoint and parameters
url = "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
params = {
    "date": "2020:2022",
    "format": "json",
    "per_page": "10000"  # To retrieve all records in one request
}

# Make the API request
response = requests.get(url, params=params)
response.raise_for_status()  # Raise an error for bad status codes

# Parse the JSON response
data = response.json()


# %%
# Final Attempt:

# First, obtain the list of all country codes provided by the World Bank API:

import requests

# API endpoint to retrieve country information
country_url = "https://api.worldbank.org/v2/country?format=json&per_page=300"

# Fetch the country data
response = requests.get(country_url)
response.raise_for_status()  # Ensure the request was successful
countries = response.json()[1]  # Extract the list of countries

# Extract country codes
country_codes = [country['id'] for country in countries if country['region']['id'] != 'NA']

# %%
# Fetch GDP Data for Each Country

import pandas as pd
import time

# Define the indicator code for GDP (current US$)
indicator_code = "NY.GDP.MKTP.CD"

# Initialize an empty DataFrame to store GDP data
gdp_data = pd.DataFrame()

# Fetch GDP data for all countries
for i in range(0, len(country_codes), 50):  # Fetch data in batches of 50 countries
    batch_codes = ';'.join(country_codes[i:i+50])
    gdp_url = f"https://api.worldbank.org/v2/country/{batch_codes}/indicator/{indicator_code}?date=2020:2022&format=json&per_page=2000"
    response = requests.get(gdp_url)
    response.raise_for_status()
    data = response.json()
    if len(data) > 1:
        batch_df = pd.json_normalize(data[1])
        gdp_data = pd.concat([gdp_data, batch_df], ignore_index=True)
    time.sleep(1)  # Pause to respect API rate limits
    

# %%
# Select relevant columns (exclude 'country.id')
gdp_data = gdp_data[['country.value', 'date', 'value']]

# Rename columns for clarity
gdp_data.columns = ['location', 'year', 'GDP (current US$)']

# Convert 'Year' to integer and 'GDP (current US$)' to numeric, handling missing values
gdp_data['year'] = gdp_data['year'].astype(int)
gdp_data['GDP (current US$)'] = pd.to_numeric(gdp_data['GDP (current US$)'], errors='coerce')

# Sort the data by 'Country Name' and 'Year' in ascending order
gdp_data = gdp_data.sort_values(by=['location', 'year']).reset_index(drop=True)

# Display the cleaned DataFrame
gdp_data.head()


# %% [markdown]
# # Merging Stringency Index and GDP Data

# %%
# Find common and unique country names between the two datasets

# Extract unique country names from each dataset
gdp_countries = set(gdp_data['location'])
stringency_countries = set(aggregated_df['location'])

# Find common countries
common_countries = gdp_countries.intersection(stringency_countries)

# Find countries unique to each dataset
unique_to_gdp = gdp_countries.difference(stringency_countries)
unique_to_stringency = stringency_countries.difference(gdp_countries)

# Display results
result = {
    "Common Countries": len(common_countries),
    "Unique to GDP Dataset": len(unique_to_gdp),
    "Unique to Stringency Dataset": len(unique_to_stringency)
}

result


# %%
# Filter the datasets to keep only the common countries
common_countries = gdp_countries.intersection(stringency_countries)

# Filter GDP data to keep only rows with common countries
filtered_gdp_data = gdp_data[gdp_data['location'].isin(common_countries)].reset_index(drop=True)

# Filter Stringency Index data to keep only rows with common countries
filtered_stringency_data = aggregated_df[aggregated_df['location'].isin(common_countries)].reset_index(drop=True)


# %%
# Merge the filtered datasets on 'location' and 'year'
merged_filtered_data = pd.merge(
    filtered_gdp_data, 
    filtered_stringency_data, 
    on=['location', 'year'], 
    how='inner'  # Ensure only common countries and years are included
)
merged_filtered_data.head()


# %% [markdown]
# ## Worldometer COVID scraping summary
# 
# Summary of data scraped (all in pandas dataframes):
# - Summary of all continents: total cases, deaths, recovered, etc. (as of 4/13/2024)
# - Summary of all countries: total cases, deaths, recovered, etc. (also links to each country's page from main page) (as of 4/13/2024)
# - Dataframes for each country (229 countries total, plus global totals): 
#     - currently infected daily
#     - total cases
#     - total deaths
#     - daily new cases
#     - daily deaths
# - A dataframe compiling the five above dataframes into averages/totals (whichever appropriate) for each country in the years 2020, 2021, and 2022
# 
# Notes about Worldometer:
# - Data stopped being reported April 13, 2024 so we have to take that into account when merging with other data sources.

# %%
# libraries
import requests
import bs4
import pandas as pd
import statistics
import warnings
warnings.filterwarnings('ignore')

# %%
result = requests.get('https://www.worldometers.info/coronavirus/')
clean_result = bs4.BeautifulSoup(result.text, 'html.parser')

# %%
tables = clean_result.find_all('table')
table = tables[0] # select 'today' table
rows = table.find_all('tr')

# %%
# extract only continents
cont_rows = rows[:9]

# %%
del cont_rows[7] # unsure what this row was, not shown on the website

# %%
data = []
# extract data for each row then add to overall dataframe
for row in cont_rows[1:]:
    cells = row.find_all('td')
    continent = cells[1].text.strip()
    total_cases = cells[2].text.strip()
    total_deaths = cells[4].text.strip()
    total_recovered = cells[6].text.strip()
    tot_cases_per_1m = cells[8].text.strip()
    tot_deaths_per_1m = cells[10].text.strip()
    population = cells[12].text.strip()
    curr_cont = [continent, total_cases, total_deaths, total_recovered, tot_cases_per_1m, tot_deaths_per_1m, population]
    data.append(curr_cont)

continents_df = pd.DataFrame(data, columns=['continent', 'total_cases', 'total_deaths', 'total_recovered', 'tot_cases_per_1m', 'tot_deaths_per_1m', 'population'])

# %% [markdown]
# Dataframe from table on main page, but focusing on the continents:

# %%
continents_df.head()

# %%
# now only countries
country_rows = rows[9:]

# %%
data = []
# extract data for each row then add to overall dataframe
for row in country_rows[:231]:
    cells = row.find_all('td')
    country = cells[1].text.strip()
    total_cases = cells[2].text.strip()
    total_deaths = cells[4].text.strip()
    total_recovered = cells[6].text.strip()
    tot_cases_per_1m = cells[8].text.strip()
    tot_deaths_per_1m = cells[10].text.strip()
    population = cells[12].text.strip()
    try : link = str(cells[1]).split('href=')[1].split('"')[1]
    except: link = ""
    else: link = str(cells[1]).split('href=')[1].split('"')[1]
    curr_cont = [country, total_cases, total_deaths, total_recovered, tot_cases_per_1m, tot_deaths_per_1m, population, link]
    data.append(curr_cont)

countries_df = pd.DataFrame(data, columns=['country', 'total_cases', 'total_deaths', 'total_recovered', 'tot_cases_per_1m', 'tot_deaths_per_1m', 'population', 'link'])

# %% [markdown]
# Dataframe from table on main page, but focusing on the countries. Also includes link for later access to each of the countries' pages

# %%
countries_df.head()

# %%
# find script elements in html source code
scripts = [str(s) for s in clean_result.find_all('script')]

# find only script elements that contained at least one chart
charts = [s for s in scripts if 'Highcharts.chart' in s]

# split by each indiv chart
charts_2 = []
for chart in charts:
    all_charts = chart.split('Highcharts.chart')
    for ch in all_charts:
        charts_2.append(ch)
charts = [s for s in charts_2 if 'chart' in s]

# get data from charts
world_df_list = []
for chart in charts[1:8]:
    title = chart.split("'")[1].replace('-', "_")
    dates = [s for s in chart.split('xAxis')[1].split('[')[1].split(']')[0].split('"') if len(s) > 2]
    
    data1 = []
    try : data1_split = chart.split('data:')[1].split('[')[1].split(']')[0].split(',')
    except: data1_split = [0 for s in dates]
    else: data1_split = chart.split('data:')[1].split('[')[1].split(']')[0].split(',')
    for d in data1_split:
        try: d_float = float(d)
        except: d_float = 0
        else: d_float = float(d)

        data1.append(d_float)

    data2 = []
    try : data2_split = chart.split('data:')[2].split('[')[1].split(']')[0].split(',')
    except: data2_split = [0 for s in dates]
    else: data2_split = chart.split('data:')[2].split('[')[1].split(']')[0].split(',')
    for d in data2_split:
        try: d_float = float(d)
        except: d_float = 0
        else: d_float = float(d)

        data2.append(d_float)
    
    df = pd.DataFrame({
        'dates': dates,
        'data1': data1,
        'data2': data2
    })
    df.columns.name = title

    world_df_list.append(df)


# %% [markdown]
# Data from the charts on the main page (different metrics by day)

# %%
world_df_list[0].head()

# %%
world_df_list[1].head()

# %%
world_df_list[2].head()

# %%
world_df_list[3].head()

# %%
world_df_list[4].head()

# %%
world_df_list[5].head()

# %%
world_df_list[6].head()

# %%
master_country_df_list = []
i = 0
for link in countries_df['link']:
    # get country name
    country = countries_df['country'][i]

    if len(link) < 1: # if link is not there, skip
        pass
    else: 
        # get request and clean
        url = 'https://www.worldometers.info/coronavirus/' + link
        result = requests.get(url)
        clean = bs4.BeautifulSoup(result.text, 'html.parser')

        # get only script elements
        scripts = [str(s) for s in clean.find_all('script')]

        # get only script elements containing charts
        charts = [s for s in scripts if 'Highcharts.chart' in s]

        # split charts
        charts_2 = []
        for chart in charts:
            all_charts = chart.split('Highcharts.chart')
            for ch in all_charts:
                charts_2.append(ch)
        charts = [s for s in charts_2 if 'chart' in s]

        # extract data from each chart
        df_list = []
        for chart in charts[1:8]:
            title = chart.split("'")[1].replace('-', "_")
            dates = [s for s in chart.split('xAxis')[1].split('[')[1].split(']')[0].split('"') if len(s) > 2]
            
            data1 = []
            try : data1_split = chart.split('data:')[1].split('[')[1].split(']')[0].split(',')
            except: data1_split = [0 for s in dates]
            else: data1_split = chart.split('data:')[1].split('[')[1].split(']')[0].split(',')
            for d in data1_split:
                try: d_float = float(d)
                except: d_float = 0
                else: d_float = float(d)

                data1.append(d_float)

            data2 = []
            try : data2_split = chart.split('data:')[2].split('[')[1].split(']')[0].split(',')
            except: data2_split = [0 for s in dates]
            else: data2_split = chart.split('data:')[2].split('[')[1].split(']')[0].split(',')
            for d in data2_split:
                try: d_float = float(d)
                except: d_float = 0
                else: d_float = float(d)

                data2.append(d_float)
            
            df = pd.DataFrame({
                'dates': dates,
                'data1': data1,
                'data2': data2
            })
            df.columns.name = title

            df_list.append(df)

        # add data to dataframe
        row = [country, df_list]
        master_country_df_list.append(row)

    i += 1

# %% [markdown]
# I made a list of nested lists in format [country, list of the data stored in the charts in a pandas df]. There are 229 countries total. Here is USA and India to demonstrate:

# %%
print(master_country_df_list[0][0])
for chart in master_country_df_list[0][1]:
    print(chart.head())

# %%
print(master_country_df_list[1][0])
for chart in master_country_df_list[1][1]:
    print(chart.head())

# %%
worldometer_avgs_df = pd.DataFrame({
    'location': [],
    'year': [],
    'total currently infected avg': [],
    'total cases': [],
    'total deaths': [],
    'daily new cases avg': [],
    'daily deaths avg': []
})

years = ['2020', '2021', '2022']
for country in master_country_df_list:
    country_name = country[0]
    for year in years:
        country_stats = [country_name, int(year)]

        # filter by year
        df_list = []
        for df in country[1]:
            year_df = df[df['dates'].str.contains(year)]
            df_list.append(year_df)

        # currently infected
        try: df = next((item for item in df_list if item.columns.name == 'graph_active_cases_total'))
        except: df = []
        else: df = next((item for item in df_list if item.columns.name == 'graph_active_cases_total'))

        if len(df) != 0:
            df = next((item for item in df_list if item.columns.name == 'graph_active_cases_total'))
            country_stats.append(statistics.mean(df['data1']))
        else:
            country_stats.append(None)

        # total cases
        try: df = next((item for item in df_list if item.columns.name == 'coronavirus_cases_log'))
        except: df = []
        else: df = next((item for item in df_list if item.columns.name == 'coronavirus_cases_log'))

        if len(df) != 0:
            df = next((item for item in df_list if item.columns.name == 'coronavirus_cases_log'))
            country_stats.append(max(df['data1']))
        else:
            country_stats.append(None)

        # total deaths
        try: df = next((item for item in df_list if item.columns.name == 'coronavirus_deaths_log'))
        except: df = []
        else: df = next((item for item in df_list if item.columns.name == 'coronavirus_deaths_log'))

        if len(df) != 0:
            df = next((item for item in df_list if item.columns.name == 'coronavirus_deaths_log'))
            country_stats.append(max(df['data1']))
        else:
            country_stats.append(None)

        # daily new cases
        try: df = next((item for item in df_list if item.columns.name == 'graph_cases_daily'))
        except: df = []
        else: df = next((item for item in df_list if item.columns.name == 'graph_cases_daily'))

        if len(df) != 0:
            df = next((item for item in df_list if item.columns.name == 'graph_cases_daily'))
            country_stats.append(statistics.mean(df['data1']))
        else:
            country_stats.append(None)
        

        # daily deaths
        try: df = next((item for item in df_list if item.columns.name == 'graph_deaths_daily'))
        except: df = []
        else: df = next((item for item in df_list if item.columns.name == 'graph_deaths_daily'))

        if len(df) != 0:
            df = next((item for item in df_list if item.columns.name == 'graph_deaths_daily'))
            country_stats.append(statistics.mean(df['data1']))
        else:
            country_stats.append(None)
        

        # add to df
        worldometer_avgs_df.loc[len(worldometer_avgs_df)] = country_stats

print(worldometer_avgs_df.head())
         



        


# %% [markdown]
# # Worldometer COVID Data with GDP and Stringency Data

# %%
# Ensure consistency in column names before merging
worldometer_avgs_df['year'] = worldometer_avgs_df['year'].astype(int)  # Ensure 'year' is integer
common_locations = set(merged_filtered_data['location']).intersection(set(worldometer_avgs_df['location']))

# Filter datasets to include only common countries
filtered_worldometer_data = worldometer_avgs_df[worldometer_avgs_df['location'].isin(common_locations)].reset_index(drop=True)
filtered_merged_data = merged_filtered_data[merged_filtered_data['location'].isin(common_locations)].reset_index(drop=True)

# Merge the datasets
final_merged_data = pd.merge(
    filtered_merged_data, 
    filtered_worldometer_data, 
    on=['location', 'year'], 
    how='inner'  # Keep only matching rows
)

# Display country name overlap summary again for reference
overlap_summary = {
    "Common Countries": len(common_locations),
    "Unique to GDP/Stringency Dataset": len(set(merged_filtered_data['location']).difference(set(worldometer_avgs_df['location']))),
    "Unique to Worldometer Dataset": len(set(worldometer_avgs_df['location']).difference(set(merged_filtered_data['location'])))
}
overlap_summary

# %%
final_merged_data.head()

# %% [markdown]
# # Exploratory Data Analysis and Visualizations
# 
# First, looking at global trends in each metric.

# %%
# libraries
import matplotlib.pyplot as plt
import numpy as np

# %%
# separate data by year
final_data_2020 = final_merged_data[final_merged_data['year'] == 2020]
final_data_2021 = final_merged_data[final_merged_data['year'] == 2021]
final_data_2022 = final_merged_data[final_merged_data['year'] == 2022]

# %%
final_data_2020.head()

# %%
def remove_outliers(data, multiplier=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return data[(data >= lower_bound) & (data <= upper_bound)]

def make_boxplot(column_name: str, multiplier=1.5):
    data = [
        remove_outliers(final_data_2020[column_name].dropna(), multiplier).tolist(),
        remove_outliers(final_data_2021[column_name].dropna(), multiplier).tolist(),
        remove_outliers(final_data_2022[column_name].dropna(), multiplier).tolist()
        ]

    # create plot
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(data, patch_artist=True)

    # colors
    colors = ['lightblue', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # labels
    ax.set_xticklabels(['2020', '2021', '2022'])
    ax.set_title(f'{column_name} by year')
    ax.set_ylabel(column_name)

    plt.show()

# %%
columns = final_data_2020.columns.tolist()[2:]

# %%
for col in columns:
    make_boxplot(col)

# %% [markdown]
# Now, to answer the first question: 
# 
# __How does the COVID-19 Stringency Index correlate with the total number of COVID-19 cases across countries?__

# %%
from scipy.stats import linregress

def remove_outliers_from_y(df, x_column, y_column, multiplier=2):
    df_clean = df.dropna(subset=[x_column, y_column])
    df_clean = df_clean[(df_clean[x_column] > 0) & (df_clean[y_column] > 0)]

    Q1 = df_clean[y_column].quantile(0.25)
    Q3 = df_clean[y_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    filtered_df = df_clean[(df_clean[y_column] >= lower_bound) & (df_clean[y_column] <= upper_bound)]
    return filtered_df

# outliers
filtered_df = remove_outliers_from_y(final_merged_data, 'stringency_index', 'total cases')

# data
x_values = filtered_df['stringency_index']
y_values = np.log(filtered_df['total cases'])

# create plot
plt.figure(figsize=(10, 7))
plt.scatter(x_values, y_values, c = 'pink', alpha = 0.6, edgecolor = 'k')

# regression line
slope, intercept, r_value, _, _ = linregress(x_values, y_values)
reg_line = slope * x_values + intercept

plt.plot(x_values, reg_line, color='red', alpha = 0.7, label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

# labels
plt.suptitle('Total Cases by Stringency Index', fontsize = 20)
plt.title(f'Regression Line: y = {slope:.3f}x + {intercept:.3f}, $R^2$ = {r_value**2:.2f}')
plt.xlabel('Stringency Index')
plt.ylabel('Log(Total Cases)')
plt.grid(True, linestyle='--', alpha = 0.5)
plt.show()

# %%
correlation_results = final_merged_data.groupby('location').apply(
    lambda group: group['stringency_index'].corr(group['total cases'])
).reset_index(name='correlation')

correlation_results.columns = ['location', 'correlation']

# figure
plt.figure(figsize=(10, 6))
plt.hist(correlation_results['correlation'].dropna(), bins=20, color='lightblue', edgecolor='k', alpha=0.7)

# labels
plt.title('Correlation Results of Stringency Index and Total Cases')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
sorted_correlations = correlation_results.sort_values(by='correlation', ascending=False).reset_index()
sorted_correlations

# %% [markdown]
# Correlation values closer to 1 indicate that as stringency index rose, total cases rose as well, and correlation values closer to -1 indicate that as stringency index rose, total cases fell. The large majority of countries showed a negative correlation between stringency index and total cases, which indicated that enforcing stricter policies led to positive outcomes in most countries. Among the strongest correlations were Singapore, Italy, and Australia, all highly developed countries.
# 
# However, some outliers had a positive correlation between stringency index and total cases, indicating that enforcing stricter policies actually led to negative outcomes in most countries. This could be due to a number of reasons, but looking at these outliers individually reveals more about these countries. The majority of countries with positive correlations are smaller, less developed countries, such as Burundi, Tonga, and Solomon Islands. The one outlier in these positively-correlated countries is China, which may be due to the fact that the outbreak originated from there.

# %% [markdown]
# Onto the next question: 
# 
# __What is the relationship between GDP and the effectiveness of COVID-19 policies in reducing mortality rates?__

# %%
# create death rate data
final_merged_data['death rate'] = final_merged_data['total deaths'] / final_merged_data['total cases']

correlation_results2 = final_merged_data.groupby('location').apply(
    lambda group: group['stringency_index'].corr(group['death rate'])
).reset_index(name='correlation')

correlation_results2.columns = ['location', 'correlation']

average_df = final_merged_data.groupby('location')[['GDP (current US$)']].mean().reset_index()

gdp_corr_df = pd.merge(average_df, correlation_results2, on='location', how='left')

gdp_corr_df_sorted = gdp_corr_df.sort_values(by='correlation', ascending=False).reset_index().dropna()

# %%
gdp_corr_df_sorted

# %%
# data
x_values = np.log(gdp_corr_df_sorted['GDP (current US$)'])
y_values = gdp_corr_df_sorted['correlation']

# figure
plt.figure(figsize=(10, 7))
plt.scatter(x_values, y_values, c = 'lightgreen', alpha = 0.6, edgecolor = 'k')

# regression line
slope, intercept, r_value, _, _ = linregress(x_values, y_values)
reg_line = slope * x_values + intercept

plt.plot(x_values, reg_line, color='lightblue', alpha = 1, label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

# labels
plt.suptitle('Effectiveness of Policies on COVID-19 Outcomes by log(GDP)', fontsize = 20)
plt.title(f'Regression Line: y = {slope:.3f}x + {intercept:.3f}, $R^2$ = {r_value**2:.2f}')
plt.xlabel('log(GDP (current US$))')
plt.ylabel('Correlation Coefficient of Stringency Index and Death Rate')
plt.grid(True, linestyle='--', alpha = 0.5)
plt.show()

# %% [markdown]
# Finally,
# 
# __How do economic disparities influence the duration and intensity of COVID-19 policy implementation across countries?__

# %%
# data
filtered_df = remove_outliers_from_y(final_merged_data, 'stringency_index', 'GDP (current US$)')

x_values = np.log(filtered_df['GDP (current US$)'])
y_values = filtered_df['stringency_index']

# figure
plt.figure(figsize=(10, 7))
plt.scatter(x_values, y_values, c = 'lightgreen', alpha = 0.6, edgecolor = 'k')

# regression line
slope, intercept, r_value, _, _ = linregress(x_values, y_values)
reg_line = slope * x_values + intercept

plt.plot(x_values, reg_line, color='lightblue', alpha = 1, label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

# labels
plt.suptitle('Policy Strictness by GDP', fontsize = 20)
plt.title(f'Regression Line: y = {slope:.3f}x + {intercept:.3f}, $R^2$ = {r_value**2:.2f}')
plt.xlabel('log(GDP (current US$))')
plt.ylabel('Stringency Index')
plt.grid(True, linestyle='--', alpha = 0.5)
plt.show()


