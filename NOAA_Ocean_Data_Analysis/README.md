**NOAA Sea Surface Temperature Data Analysis**

The National Oceanographic and Atmospheric Association, NOAA, has dedicated scientists and advanced technology to provide reliable information about changes in climate, weather, oceans and coasts. Such information can be found online and it's available to the public. The NOAA dataset 'International Comprehensive Ocean-Atmosphere Data Set ICOADS' was used for this project. The dataset can be found in Google Cloud's [Marketplace](https://console.cloud.google.com/marketplace), and it has the following description:

> "The ICOADS dataset contains global marine data from ships (merchant, navy, research) and buoys, each capturing details according to the current weather or ocean conditions (wave height, sea temperature, wind speed, and so on). Each record contains the exact location of the observation which is great for visualizations. The historical depth of the data is quite comprehensive — there are records going back to 1662.
This public dataset is hosted in Google BigQuery and is included in BigQuery's 1TB/mo of free tier processing. This means that each user receives 1TB of free BigQuery processing every month, which can be used to run queries on this public dataset." 

Google BigQuery allows the database to be explored and manipulated using SQL queries. In this case, we want to explore the average sea surface temperature for the duration of 67 years, extending from the year 1950 to the year 2017. The main objective is to visualize the changes in sea surface temperature as a function of time.

Project Methods:

- Query the database using SQL to extract the data
- Export the results of the query into a readable Python format.
- Visualize important features in the data.
- Use machine learning methods to predict future sea surface temperatures.
