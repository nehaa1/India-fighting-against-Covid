# India-fighting-against-Covid # 

This project depicts the fight of India against Covid-19. 
The datasets are taken from various sources made available by Government of India as as worldwide data.

Various factors that are considered while working on this project are-
1. Where does India stand at present?
2. Is healthcare in India good enough to handle this situation on large scale?
3. Does population density has anything to do with spread of disease?

Before proceeding you need to install pycountry_convert, googlemaps and plotly if haven't used it before.
Going stepwise first perform countrywise analysis and then statewise to get bigger picture. Also analyse the healthcare facility in each state and over all India.

COVID-19 cases at daily level is present in covid_19_india.csv file.
Individual level details are present in IndividualDetails.csv file.
Population at state level is present in population_india_census2011.csv file.
Number of COVID-19 tests at daily level in ICMRTestingDetails.csv file.
Number of hospital beds in each state in present in HospitalBedsIndia.csv file.

Main file in this dataset is covid_19_data.csv and the detailed descriptions are below.

covid_19_data.csv

Sno - Serial number

ObservationDate - Date of the observation in MM/DD/YYYY

Province/State - Province or state of the observation (Could be empty when missing)

Country/Region - Country of observation

Last Update - Time in UTC at which the row is updated for the given province or country. (Not standardised and so please clean before using it)

Confirmed - Cumulative number of confirmed cases till that date

Deaths - Cumulative number of of deaths till that date

Recovered - Cumulative number of recovered cases till that date
