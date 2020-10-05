# India fighting against Covid # 

COVID 19 has spread all across the world so swiftly that it was impossible to take measures against it. 
Gradually it started spreading across the globe. In India the outbreak reached a bit late (in March 2020) as compared to the other countries, 
but it started to pick up pace within a short duration.

This project depicts the fight of India against Covid-19. 
The datasets are taken from various sources made available by Government of India as as worldwide data.

## Installation
* Install pycountry_convert
* googlemaps 
* and plotly if haven't used it before.

## Related Blog
Complete analysis is conveyed via following blog with appropriate plots.

https://medium.com/@nehaa.139/why-is-india-struggling-to-fight-against-covid-19-819c9a9eec53


## About the Datasets
The number of new cases are increasing day by day all around the world. This dataset has updated information from the states and union territories of India at daily level.

State level data comes from <b> Ministry of Health & Family Welfare </b>.

Individual level data comes from <b> covid19india </b>.

Main file in this dataset is covid_19_data.csv which contains the data across the globe.
* COVID-19 cases at daily level is present in covid_19_india.csv file.
* Individual level details are present in IndividualDetails.csv file obtained from http://portal.covid19india.org/.
* Population at state level is present in population_india_census2011.csv file.
* Number of COVID-19 tests at daily level are present in ICMRTestingDetails.csv file.
* Number of hospital beds in each state are present in HospitalBedsIndia.csv file taken from https://pib.gov.in/PressReleasePage.aspx?PRID=1539877.
* Travel history dataset is taken from https://www.kaggle.com/dheerajmpai/covidindiatravelhistory

## Analysis using available data

To come across this scenario, we need to answer few questions. Here are those questions:
1. Where does India stand at present?
2. What is the effect of doubling rate?
3. Is healthcare in India good enough to handle this situation on large scale?
4. Does population density has anything to do with spread of this disease?

Going stepwise first perform countrywise analysis to get the current ranking of India across the globe.
Studying statewise to get bigger picture I have plotted the number of cases reported daily in each state, calculated doubling rate, effect of population density etc. Also analyse the healthcare facility in each state and over all India. 

Using all this data, I have attempted to analyse the data as much as I could. Analysis shows the India's ranking in Asiaas well as the world. Also I have studied the doubling rate, testing labs facility in India, measures taken by India and various factors affecting the spread of Covid-19.

## Acknowledgements
Thanks to <b> Indian Ministry of Health & Family Welfare </b> for making the data available to general public.

Thanks to <b> covid19india.org </b> for making the individual level details and testing details available to general public.
