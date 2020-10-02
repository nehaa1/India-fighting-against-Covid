#!/usr/bin/env python
# coding: utf-8

# ## <a> Analysis of COVID-19 in India ##

# ### Downloding and Installing Prerequisite ###
# 
# * pip install pycountry_convert 
# * pip install GoogleMaps

# In[ ]:


get_ipython().system('pip install pycountry_convert')


# In[ ]:


get_ipython().system('pip install GoogleMaps')


# ### Imports and Datasets ### 
# Import following libraries.
# * Pandas - for dataset handeling
# * Numpy - Support for Pandas and calculations
# * Matplotlib - for visualization (Platting graphas)
# * pycountry_convert - Library for getting continent (name) to from their country names
# * plotly - for interative plots

# In[22]:


import plotly
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import googlemaps
import re 
import pycountry
import pycountry_convert as pc
import requests
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# ### India's current ranking in the world and Asia 

# In[23]:


# Defining Function for getting continent and code country.

class country_utils():
    def __init__(self):
        self.d = {}
    
    def get_dic(self):
        return self.d
    
    def get_country_details(self,country):
        """Returns country code(alpha_3) and continent"""
        try:
            country_obj = pycountry.countries.get(name=country)
            if country_obj is None:
                c = pycountry.countries.search_fuzzy(country)
                country_obj = c[0]
            continent_code = pc.country_alpha2_to_continent_code(country_obj.alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj.alpha_3, continent
        except:
            if 'Congo' in country:
                country = 'Congo'
            if country == 'Mainland China':
                country = 'China'
            elif country == 'Diamond Princess' or country == 'Laos' or country == 'MS Zaandam'            or country == 'Holy See' or country == 'Timor-Leste':
                return country, country
            elif country == 'Korea, South' or country == 'South Korea':
                country = 'Korea, Republic of'
            elif country == 'Taiwan*':
                country = 'Taiwan'
            elif country == 'Burma':
                country = 'Myanmar'
            elif country == 'West Bank and Gaza':
                country = 'Gaza'
            else:
                return country, country
            country_obj = pycountry.countries.search_fuzzy(country)
            continent_code = pc.country_alpha2_to_continent_code(country_obj[0].alpha_2)
            continent = pc.convert_continent_code_to_continent_name(continent_code)
            return country_obj[0].alpha_3, continent
    
    def get_iso3(self, country):
        '''Returns country code'''
        return self.d[country]['code']
    
    def get_continent(self,country):
        '''Returns continent'''
        return self.d[country]['continent']
    
    def add_values(self,country):
        self.d[country] = {}
        self.d[country]['code'],self.d[country]['continent'] = self.get_country_details(country)
    
    def fetch_iso3(self,country):
        if country in self.d.keys():
            return self.get_iso3(country)
        else:
            self.add_values(country)
            return self.get_iso3(country)
        
    def fetch_continent(self,country):
        if country in self.d.keys():
            return self.get_continent(country)
        else:
            self.add_values(country)
            return self.get_continent(country)


# In[24]:


# Importing the Covid-19 data taken on daily basis level worldwide

df_world = pd.read_csv('dataset/covid_19_data.csv')
df_world.ObservationDate = pd.to_datetime(df_world.ObservationDate, format='%m/%d/%Y')
max_date = df_world.ObservationDate.max()
df_world = df_world[df_world.ObservationDate==max_date]

# Changing the name of column for better practice
df_world.rename(columns={'Country/Region':'Country'},inplace=True)

# Creating copy of Covid-19 datset file
df_cont = df_world.copy() 

# Seggregating the data according to confirmed, deaths and recovered cases in India
df_world = df_world.groupby(['Country'],as_index=False)['Confirmed','Deaths','Recovered'].sum()
df_world['rank_c'] = df_world['Confirmed'].rank(ascending=False)
df_world['rank_d'] = df_world['Deaths'].rank(ascending=False)
df_world['rank_r'] = df_world['Recovered'].rank(ascending=False)
world_stat = (df_world.loc[df_world['Country']=='India'])
world_stat.set_index('Country',inplace=True)
world_stat = world_stat.astype(int)


# In[25]:


# Function to get ranking of India in the world and Asia according to cases noted on daily basis
obj = country_utils()
df_cont['continent'] = df_cont.apply(lambda x: obj.fetch_continent(x['Country']), axis=1)
df_cont = df_cont.groupby(['continent','Country'],as_index=False)['Confirmed','Deaths','Recovered'].sum()
df_cont = df_cont[df_cont['continent']=='Asia']

# Ranking is calculated for confirmed cases, deaths and recoveries
df_cont['rank_c'] = df_cont['Confirmed'].rank(ascending=False)
df_cont['rank_d'] = df_cont['Deaths'].rank(ascending=False)
df_cont['rank_r'] = df_cont['Recovered'].rank(ascending=False)
cont_stat = (df_cont.loc[df_cont['Country']=='India'])
cont_stat.set_index('Country',inplace=True)
cont_stat.drop('continent',inplace=True,axis=1)
cont_stat = cont_stat.astype(int)


# The table below shows the current figures and ranking for India in the World and in Asia.

# In[26]:


# Table to indicate ranking of India as of the date of analysis

def make_538(fig,title=None,xtext=None,ytext=None,hovermode='x',width=700,height=400,margin=dict(t=50,b=10,l=10,r=10),
             legend=None,annotations=None):
    fig.update_layout(
        template='simple_white',
        title=title,
        hovermode=hovermode,
        xaxis=dict(title=xtext,showline=False,showgrid=True,ticks='',gridcolor=colors['grid']),
        yaxis=dict(title=ytext,showline=False,showgrid=True,ticks='',gridcolor=colors['grid']),
        paper_bgcolor=colors['bg'],
        plot_bgcolor=colors['bg'],
        width=width,
        height=height,
        margin=margin,
        legend=legend,
        annotations=annotations
    )
    return fig

# Color pallette for the table
colors = dict(bg='#f0f0f0',
              grid='#d7d7d7',
              cases='#30a2da',
              deaths='#fc4f30',
              recoveries='#6d904f',
              shape='#8b8b8b')

# Importing function from library for better visualization
import plotly.graph_objects as go

values = [['Figures','World Ranking','Asia Ranking'],
          [world_stat.Confirmed['India'],world_stat.rank_c['India'],cont_stat.rank_c['India']],
          [world_stat.Deaths['India'],world_stat.rank_d['India'],cont_stat.rank_d['India']],
          [world_stat.Recovered['India'],world_stat.rank_r['India'],cont_stat.rank_r['India']]]

# Defining cell parameters in the table
fig = go.Figure(data=[go.Table(
  columnorder = [1,2,3,4],
  columnwidth = [300,300],
  header = dict(
    values = [['<b>STATISTICS</b><br>as of '+str(max_date.day)+' '+ max_date.month_name()],
              ['<b>CASES</b>'],['<b>DEATHS</b>'],['<b>RECOVERIES</b>']],
    line_color=colors['grid'],
    fill_color=colors['cases'],
    align=['left','center'],
    font=dict(color='white', size=13),
    height=40
  ),
  cells=dict(
    values=values,
    line_color=colors['grid'],
    fill=dict(color=[colors['shape'], 'white']),
    align=['left', 'center'],
    font_size=13,
    height=35)
    )
])
fig = make_538(fig,height=250,title=dict(text='<b>Report from Ground Zero</b>',font=dict(family='Helvetica')))
fig.show()


# In[27]:


# Importing the Covid-19 data taken on daily basis level in India

df = pd.read_csv('dataset/covid_19_india.csv')
df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%y')
df['Date'] = df['Date'].dt.date

# Changing the name of column as per convenience
df.rename(columns={'Date':'date','State/UnionTerritory':'state','ConfirmedIndianNational':'confirmed_in',                   'ConfirmedForeignNational':'confirmed_fr'}, inplace=True)

# Dropping unnecessary data
df.drop(['Sno','Time'],axis=1,inplace=True)
df['state'] = df.apply(lambda x: 'Nagaland' 
                       if x['state']=='Nagaland#' else 'Jharkhand' if x['state']=='Jharkhand#' else x['state'], axis=1)
df = df[df['state']!='Unassigned']
df.reset_index(inplace=True)


# In[28]:


# Defining a function to plot confirmed cases, deaths and recoveries on a daily basis

def add_daily_measures(df):
    has_state=False
    if 'state' in df.columns:
        states = []
        has_state = True
    df.loc[0,'Daily Cases'] = df.loc[0,'Confirmed']
    df.loc[0,'Daily Deaths'] = df.loc[0,'Deaths']
    df.loc[0,'Daily Cured'] = df.loc[0,'Cured']
    for i in range(1,len(df)):
        if has_state:
            if df.loc[i,'state'] in states:
                df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed'] - df.loc[i-1,'Confirmed']
                df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths'] - df.loc[i-1,'Deaths'] 
                df.loc[i,'Daily Cured'] = df.loc[i,'Cured'] - df.loc[i-1,'Cured']
            else:
                states.append(df.loc[i,'state'])
                df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed']
                df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths']
                df.loc[i,'Daily Cured'] = df.loc[i,'Cured']
        else:
            df.loc[i,'Daily Cases'] = df.loc[i,'Confirmed'] - df.loc[i-1,'Confirmed']
            df.loc[i,'Daily Deaths'] = df.loc[i,'Deaths'] - df.loc[i-1,'Deaths'] 
            df.loc[i,'Daily Cured'] = df.loc[i,'Cured'] - df.loc[i-1,'Cured']
            
    '''Making the first row as 0 because we don't know the previous value'''
    
    df.loc[0,'Daily Cases'] = 0
    df.loc[0,'Daily Deaths'] = 0
    df.loc[0,'Daily Cured'] = 0
    return df


# In[29]:


df.loc[1428,'state'] = 'Madhya Pradesh'
df.loc[1428,'Deaths'] = '119'
df.fillna(0,inplace=True)
df.loc[df.Deaths=='0#','Deaths'] = 0
df.Deaths = df.Deaths.astype(np.int16)


# The plot below shows the total cases, deaths and recoveries reported on a daily basis. The worrying point is that even after almost nation-wide lockdowns the case are increasing.

# In[30]:


# Important dates showing the lockdwon phases and unlock phases  
imp_dates = [dict(date='2020-03-23',event="Lockdown Phase 1<br><b>23<sup>rd</sup> March</b>"),
             dict(date='2020-04-15',event="Lockdown Phase 2<br><b>15<sup>th</sup> April</b>"),
             dict(date='2020-05-04',event="Lockdown Phase 3<br><b>4<sup>th</sup> May</b>"),
             dict(date='2020-05-18',event="Lockdown Phase 4<br><b>18<sup>th</sup> May</b>"),
             dict(date='2020-06-01',event="Unlock 1.0<br><b>1<sup>st</sup> June</b>"),
             dict(date='2020-07-01',event="Unlock 2.0<br><b>1<sup>st</sup> July</b>"),
             dict(date='2020-08-01',event="Unlock 3.0<br><b>1<sup>st</sup> August</b>")]

# Plotting the cases recovered, confirmed and deaths in India 
df_ind = df.copy()
df_ind = df_ind.groupby('date',as_index=False)['Cured','Deaths','Confirmed'].sum()
df_ind = add_daily_measures(df_ind)

fig = go.Figure(data=[
    go.Bar(name='Deaths', x=df_ind['date'], y=df_ind['Daily Deaths'],marker_color='crimson',marker_line_color='black'),
    go.Bar(name='Recoveries', x=df_ind['date'], y=df_ind['Daily Cured'],marker_color='limegreen',marker_line_color='black'),
    go.Bar(name='Cases', x=df_ind['date'], y=df_ind['Daily Cases'],marker_color='royalblue',marker_line_color='black'),
])

annotations = []

# Including the above mentioned important dates in the plot below
for date in imp_dates:
    fig.add_shape(type='line',xref='x',yref='y',layer='below',
                  x0=date['date'] ,y0=0,x1=date['date'],y1=140000,
                  line=dict(dash='dot',color=colors['shape'],width=3))
    
    annotations.append(dict(x=date['date'], y=120000, xref="x", yref="y",textangle=-45, 
                            text=date['event'], font=dict(size=10), showarrow=False))
    

legend=dict(orientation='h',x=0.5,y=1.1,bgcolor=colors['bg'])
fig.update_layout(template="simple_white",barmode='relative', title='Total Cases, Deaths & Recoveries',
                  legend= legend, annotations=annotations)
fig.show()


# In[31]:


get_ipython().run_cell_magic('HTML', '', '<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/2061549" data-url="https://flo.uri.sh/visualisation/2061549/embed" aria-label=""><script src="https://public.flourish.studio/resources/embed.js"></script></div>')


# In[32]:


# Code to get the dataset for Bar chart race
df_br = df.copy()
df_br = df_br.pivot(index='state',columns='date',values='Confirmed')
df_br.fillna(0,inplace=True)
df_br.reset_index(level=0, inplace=True)
df_br.columns.name = ''
df_br.to_csv(r'confirmed_cases_india.csv',index=False)


# ###  Studying Doubling rates 

# In[33]:


# Function to study the doubling rate in India over a period of time

def doubling_rate_india(x):
    '''Equation to calculate doubling rate'''
    dr = np.log(2)/np.log(x['Confirmed']/x['Confirmed_prev'])
    return np.round(dr,2)

df_ind['Confirmed_prev'] = df_ind['Confirmed'].shift(1)
df_ind['Doubling rate'] = df_ind.apply(lambda x: doubling_rate_india(x),axis=1)

# Plot to show doubling rate variation over time
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ind['date'],y=df_ind['Doubling rate']))
fig.update_layout(template='presentation',title="India's Doubling rate variation over Time")
fig.show()


# In[34]:


# Calculating doubling rate for each state 

doubling_rate = 0
state = df.state.unique().tolist()[0]
def calc_doubling_rate(x):
    global growth_rate
    global state
    
    '''Calculating doubling rate considering each state'''
    
    if x['state']!=state:
        doubling_rate=0
        state = x['state']
    try:
        dr = np.log(2)/np.log(x['Confirmed']/x['Confirmed_prev'])
    except ZeroDivisionError:
        dr = 0
    return np.round(dr,2)

df_dr = df.copy()
df_dr.sort_values(['state','date'],inplace=True)
df_dr.reset_index(drop=True,inplace=True)
df_dr['Confirmed_prev'] = df_dr.groupby('state')['Confirmed'].transform(lambda x: x.shift(1))
df_dr['Doubling rate'] = df_dr.apply(lambda x: calc_doubling_rate(x),axis=1)
df_curr_dr = df_dr[df_dr['date']==df_dr.date.max()]

'''Sorting states according to confirmed cases'''
df_curr_dr.sort_values('Confirmed',ascending=False,inplace=True) 
df_curr_dr=df_curr_dr[['state','Confirmed','Doubling rate']]
df_curr_dr=df_curr_dr.nlargest(15,'Confirmed')
df_curr_dr.reset_index(drop=True,inplace=True)

# Set CSS properties for th elements in dataframe
th_props = [
  ('font-size', '12px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7f7f9')
  ]

# Set CSS properties for td elements in dataframe
td_props = [
  ('font-size', '12px')
  ]

# Set table styles
styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

(df_curr_dr.style
 .background_gradient(cmap='Reds_r',subset='Doubling rate')
 .background_gradient(cmap='Blues',subset='Confirmed')
 .set_caption('<b>Doubling Rates of 15 Indian States with most cases')
 .set_table_styles(styles))


# In[15]:


# Showing total cases per state
df_map = df.copy()
def change_state_name(state):
    '''Changing the state names as per latest terminology given''' 
    if state == 'Odisha':
        return 'Orissa'
    elif state == 'Telengana':
        return 'Telangana'
    elif state == 'Andaman and Nicobar Islands':
        return 'Andaman and Nicobar'
    return state
r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson')
geojson = r.json()

df_map['state'] = df_map.apply(lambda x: change_state_name(x['state']), axis=1)
last_date = df_map.date.max()
df_map = df_map[df_map['date']==last_date]

fig = px.choropleth_mapbox(df_map, geojson=geojson, color="Confirmed",
                    locations="state", featureidkey="properties.NAME_1",
                    hover_name='state',
                    hover_data=['Cured','Deaths'],
                    center={"lat": 20.5937, "lon": 78.9629},
                    mapbox_style="carto-positron",
                    zoom=2.75,
                    color_continuous_scale=px.colors.qualitative.Vivid,
                    title='Total Cases per State'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.show()


# In[35]:


# Dataset showcasing individual details in each district
df_ind_det = pd.read_csv('dataset/IndividualDetails.csv')
r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson')
geojson = r.json()

df_dist = df_ind_det.groupby('detected_district',as_index=False)['id'].count()
df_dist.rename(columns={'detected_district':'District','id':'Cases Reported'},inplace=True)
fig = px.choropleth_mapbox(df_dist, geojson=geojson, color="Cases Reported",
                    locations="District", featureidkey="properties.NAME_2",
                    hover_name='District',
                    center={"lat": 20.5937, "lon": 78.9629},
                    mapbox_style="carto-positron",
                    zoom=2.75,
                    color_continuous_scale=px.colors.qualitative.Vivid,
                    title='Total Cases per District'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig = make_538(fig,title='Total Cases per District')
fig.show()


# In[ ]:


# Importing dataset having individual level details

df_ind_det = pd.read_csv('dataset/IndividualDetails.csv')

#Importing district data 
r = requests.get(url='https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson')
geojson = r.json()
df_dist = df_ind_det.groupby('detected_district',as_index=False)['id'].count()

# Renaming the colomns
df_dist.rename(columns={'detected_district':'District','id':'Cases Reported'},inplace=True)

# Plotting the total cases per district
fig = px.choropleth_mapbox(df_dist, geojson=geojson, color="Cases Reported",
                    locations="District", featureidkey="properties.NAME_2",
                    hover_name='District',
                    center={"lat": 20.5937, "lon": 78.9629},
                    mapbox_style="carto-positron",
                    zoom=2.75,
                    color_continuous_scale=px.colors.qualitative.Vivid,
                    title='Total Cases per District'
                   )
fig.update_geos(fitbounds="locations", visible=True)
fig.update_geos(projection_type="orthographic")
fig.show()


# In[ ]:


# Importing the dataset containing district level data showing number of zones
df_zone = pd.read_csv('dataset/India-District-Zones.csv')

'''Sorting the districts in each state according to the zones assigned as red, orange 
  and green zone according to the number of cases observed'''

df_zone = df_zone.groupby(['State','Zone'],as_index=False)['District'].count()
df_zone.sort_values('District',inplace=True)

fig = go.Figure()
df_g = df_zone[df_zone['Zone']=='Green Zone']
fig.add_trace(go.Bar(name='Green zone',x=df_g['State'],y=df_g['District'], marker_color='Green'))
df_o = df_zone[df_zone['Zone']=='Orange Zone']
fig.add_trace(go.Bar(name='Orange zone',x=df_o['State'],y=df_o['District'], marker_color='Orange'))
df_r = df_zone[df_zone['Zone']=='Red Zone']
fig.add_trace(go.Bar(name='Red zone',x=df_r['State'],y=df_r['District'], marker_color='Red'))

fig.update_layout(title='<b>Number of Zones per State from 4th May</b><br>Specific to Lockdown 3.0',
                  legend=legend, barmode='stack')
fig.show()


# ###  Testing in India

# In[ ]:


# Importing dataset containing Testing labs in India
df_lab = pd.read_csv('dataset/ICMRTestingLabsWithCoords.csv')

# Plotting testing labs on the map of India
fig = px.scatter_mapbox(df_lab,
                        lat="latitude",
                        lon="longitude",
                        mapbox_style='streets',
                        hover_name='lab',
                        hover_data=['city','state','pincode'],
                        zoom=2.5,
                        size_max=15,
                        title= 'COVID19 Testing Labs in India')
fig.show()


# In[ ]:


def add_text(ax,fontsize=12):
    '''Showcasing the number of Covid-19 testing labs per state/union teritorry'''
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{}'.format(int(y)), (x.mean(), y), ha='center', va='bottom',size=fontsize)
        
plt.figure(figsize=(16,6))
ax = sns.countplot(data=df_lab,x='state',color='salmon',order = df_lab['state'].value_counts().index)
add_text(ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right",size=12)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Number of COVID-19 testing labs per State/Union Territory')
plt.tight_layout()
plt.show()


# In[ ]:


# Importing the test details conducted in each state
df_tes_st = pd.read_csv('dataset/StatewiseTestingDetails.csv')
fig = go.Figure()
for state in df_tes_st.State.unique():
    '''Counting number of tests conducted in each state'''
    df_s = df_tes_st[df_tes_st['State']==state]
    fig.add_trace(go.Scatter(x=df_s['Date'],y=df_s['TotalSamples'],mode='lines',name=state))
    #fig.add_trace(go.Bar(x=df_s['Date'],y=df_s['Positive']))
    
fig.update_layout(template='plotly_white',title='State-wise number of tests',
                  xaxis=dict(range=[df_tes_st.Date.min(),df.date.max()]))
fig.show()


# ###  Statewise Analysis

# In[ ]:


# Getting the datasets containing details on population in India, number of hosptital beds in India, 
# testing details in each state to study overall statistical analysis

df_pop = pd.read_csv('dataset/population_india_census2011.csv')
df_bed = pd.read_csv('dataset/HospitalBedsIndia.csv')
df_test = pd.read_csv('dataset/StatewiseTestingDetails.csv',parse_dates=True)
df_bed['Total beds'] = df_bed.NumPublicBeds_HMIS + df_bed.NumRuralBeds_NHP18 + df_bed.NumUrbanBeds_NHP18


# In[ ]:


def get_area(s):
    if pd.isnull(s):
        return s
    temp = re.findall(r'\d+.?\d+', s)
    temp = [x.replace(',','') for x in temp]
    res = list(map(int, temp))
    return res

def get_density(s):
    if pd.isnull(s):
        return s
    temp = re.findall(r'\d+[.\.]?\d+', s)
    temp = [x.replace(',','') for x in temp]
    res = list(map(float, temp))
    return res


# In[ ]:


# Fetching the deatils as per latest date
df_latest = df[df['date']==df.date.max()]

# Merging the data of states and union territories
df_latest = df_latest.merge(df_pop,how='left',left_on='state',right_on='State / Union Territory')
df_latest = df_latest.merge(df_bed,how='left',left_on='state',right_on='State/UT')

# Getting the neccessary details required to study the curent scenario in India
df_latest = df_latest[['state','Cured','Deaths','Confirmed','Population','Rural population',
                       'Area','Density','Gender Ratio','Total beds','TotalPublicHealthFacilities_HMIS']]
df_latest = df_latest[df_latest['state']!='Cases being reassigned to states']

# Dropping unnecessary columns
df_latest.drop([7,30],inplace=True)

# Defining function to get area and density details
df_latest['Area(sq km)'] = df_latest.apply(lambda x:get_area(x['Area'])[0],axis=1)
df_latest['Area(sq miles)'] = df_latest.apply(lambda x:get_area(x['Area'])[1],axis=1)
df_latest['Density(sq km)'] = df_latest.apply(lambda x:get_density(x['Density'])[0],axis=1)
df_latest['Density(sq miles)'] = df_latest.apply(lambda x:get_density(x['Density'])[1],axis=1)
df_latest.drop(['Area','Density'],axis=1,inplace=True)
df_latest.rename(columns={'state':'State'},inplace=True)

# Calculating number cases/million population
df_latest['Cases/million'] = round((df_latest.Confirmed/df_latest.Population)*1000000).astype(int)
df_latest.fillna(0,inplace=True)

# Number of beds/million
df_latest['Beds/million'] = round((df_latest['Total beds']/df_latest.Population)*1000000).astype(int)

# Health facilities/100 sq.km area
df_latest['Health Facilities/100sq.km'] = round((df_latest['TotalPublicHealthFacilities_HMIS']/df_latest['Area(sq km)'])
                                                *1000).astype(int)

# Mortality rate in percentage
df_latest['Mortality Rate %'] = np.round((df_latest.Deaths/df_latest.Confirmed)*100,2)

# Recovery rate in percentage
df_latest['Recovery Rate %'] = np.round((df_latest.Cured/df_latest.Confirmed)*100,2)

# Arranging table according to number of confirmed cases
df_latest.sort_values('Confirmed',ascending=False,inplace=True)
df_latest.reset_index(inplace=True,drop=True)
df_table = df_latest[['State','Cured','Deaths','Confirmed','Mortality Rate %','Recovery Rate %',
                      'Cases/million','Beds/million','Health Facilities/100sq.km']]
(df_table.style
 .background_gradient(cmap='Blues',subset=['Confirmed','Cases/million','Beds/million','Health Facilities/100sq.km'])
 .background_gradient(cmap='Greens', subset=['Cured','Recovery Rate %'])
 .background_gradient(cmap='Reds', subset=['Deaths','Mortality Rate %'])
 .set_caption('<b>COVID19: Statistics about India')
 .set_table_styles(styles))


# ### Effect of population density on the spread of Covid-19 in India
# 

# In[ ]:


# Studying the effect of popolation density on spread of Covid-19
rows = 1
cols = 3
fig = make_subplots(rows=rows,cols=cols,shared_yaxes=True,horizontal_spacing=0.03,
                    subplot_titles=['Cases','Deaths','Recoveries'])
d_cols = {1:'Confirmed',2:'Deaths',3:'Cured'}
colors = plotly.colors.DEFAULT_PLOTLY_COLORS
for c in range(1,4):
    for i,state in enumerate(df_latest.State.unique().tolist()):
        '''Getting population and density details to calculate population density for confirmed cases, deaths and recoveries.'''
        group = df_latest[df_latest['State']==state]
        fig.append_trace(go.Scatter(
            name = state,
            x = group[d_cols[c]],
            y = group['Density(sq km)'],
            mode = 'markers',
            marker = dict(
                color = colors[i%len(colors)],
                size = group['Population']/(5*10**6)
            ),
            legendgroup = state,
            showlegend = True if c==1 else False,
            text= '<b>'+state.upper()+'</b><br>'+
                  '<b>Population: </b>'+ group['Population'].astype(str) +'<br>'+
                  '<b>'+d_cols[c]+': </b>'+ group[d_cols[c]].astype(str) +'<br>'+
                  '<b>Pop. Density: </b>'+ group['Density(sq km)'].astype(str) +'<br>',
            hoverinfo = 'text',
        ),1,c)
        fig.update_xaxes(tickfont=dict(size=8),row=1,col=c)
fig.update_yaxes(title_text='Population Density(per sq km)',row=1,col=1)
fig.update_layout(width=700,height=400,title="States' Population Density Analysis",
                 margin=dict(t=70,b=0,r=10,l=0), 
                 legend=dict(title='State/Union Territory',font=dict(size=8))) 
fig.show()


# In[ ]:


df_states = df.copy()
def add_days(df,new_col,basis):
    '''Studying the cumulative trend since the day it was first reported'''
    states = {}
    df[new_col] = 0
    for i in range(len(df_states)):
        if df_states.loc[i,'state'] in states:
            df_states.loc[i,new_col] = (df_states.loc[i,'date'] - states[df_states.loc[i,'state']]).days
        else:
            if df_states.loc[i,basis] > 0:
                states[df_states.loc[i,'state']] = df_states.loc[i,'date']
    return df
df_states = add_days(df_states,'day_since_inf','Confirmed')
df_states = add_days(df_states,'day_since_death','Deaths')
df_states = add_days(df_states,'day_since_cure','Cured')


# In[ ]:


rows = 1
cols = 3
fig = make_subplots(rows=rows,cols=cols,
                    subplot_titles=['Cases','Deaths','Recoveries'])
d_cols = {1:'Confirmed',2:'Deaths',3:'Cured'}
colors = plotly.colors.DEFAULT_PLOTLY_COLORS
for c in range(1,4):
    for i,state in enumerate(df_states.state.unique().tolist()):
        group = df_states[df_states['state']==state]
        fig.append_trace(go.Scatter(
            name = state,
            y = group[d_cols[c]],
            x = group['day_since_inf'],
            mode = 'lines',
            line = dict(
                color = colors[i%len(colors)],
            ),
            legendgroup = state,
            showlegend = True if c==1 else False,
            text= '<b>'+state.upper()+'</b><br>'+
                   '<b>Day: </b>'+ group['day_since_inf'].astype(str) +'<br>'+
                   '<b>'+d_cols[c]+': </b>'+ group[d_cols[c]].astype(str) +'<br>',
            hoverinfo = 'text',
        ),1,c)
        fig.update_xaxes(tickfont=dict(size=8),row=1,col=c)
        fig.update_yaxes(tickfont=dict(size=8),row=1,col=c)
fig.update_yaxes(title_text='Confirmed Numbers',row=1,col=1)
fig.update_xaxes(title_text='Days since first infection was reported',row=1,col=2)
fig.update_layout(width=700,height=400,
                title="<b>States' Trend Analysis</b><br>Cumulative Statistics over time",
                 margin=dict(t=100,b=0,r=10,l=0), 
                 legend=dict(title='State/Union Territory',font=dict(size=8))) 
fig.show()


# In the below three plots, I'm plotting the 7-day rolling mean of daily cases, deaths and recoveries reported against the day it's first instance was reported.

# In[ ]:


# Getting details of each state on daily basis to study the cumulative trend for 7-day rolling
df_states.sort_values(by=['state','date'],inplace=True)
df_states.reset_index(inplace=True,drop=True)
df_states_daily = add_daily_measures(df_states)
df_states_daily.fillna(0,inplace=True)


# In[ ]:


states = df_states_daily['state'].unique().tolist()
df_roll = pd.DataFrame()
for state in states:
    '''Collecting data to plot 7-day rolling mean of daily cases, deaths and recoveries reported 
    against the day it's first instance was reported'''
    
    df_state = df_states_daily[df_states_daily['state']==state]
    '''Calculating avarage to plot 7-day rolling'''
    df_state['roll_avg_c'] = np.round(df_state['Daily Cases'].rolling(7).mean())
    df_state['roll_avg_d'] = np.round(df_state['Daily Deaths'].rolling(7).mean())
    df_state['roll_avg_r'] = np.round(df_state['Daily Cured'].rolling(7).mean())
    df_roll = df_roll.append(df_state,ignore_index=True)


# In[ ]:


rows = 1
cols = 3
fig = make_subplots(rows=rows,cols=cols,
                    subplot_titles=['Cases','Deaths','Recoveries'])
d_cols = {1:'roll_avg_c',2:'roll_avg_d',3:'roll_avg_c'}
colors = plotly.colors.DEFAULT_PLOTLY_COLORS
for c in range(1,4):
    for i,state in enumerate(df_roll.state.unique().tolist()):
        '''Fetching details of confirmed cases, deaths and recoveries calculated in the above cell to make a plot'''
        group = df_roll[df_roll['state']==state]
        fig.append_trace(go.Scatter(
            name = state,
            y = group[d_cols[c]],
            x = group['day_since_inf'],
            mode = 'lines',
            line = dict(
                color = colors[i%len(colors)],
            ),
            legendgroup = state,
            showlegend = True if c==1 else False,
            text= '<b>'+state.upper()+'</b><br>'+
                   '<b>Day: </b>'+ group['day_since_inf'].astype(str) +'<br>'+
                   '<b>'+d_cols[c]+': </b>'+ group[d_cols[c]].astype(str) +'<br>',
            hoverinfo = 'text',
        ),1,c)
        fig.update_xaxes(tickfont=dict(size=8),row=1,col=c)
        fig.update_yaxes(tickfont=dict(size=8),row=1,col=c)
fig.update_yaxes(title_text='Confirmed Numbers(7-Day Rolling average)',
                 title_font_size=10,row=1,col=1)
fig.update_xaxes(title_text='Days since first infection was reported',row=1,col=2)
fig.update_layout(width=700,height=400,
                title="<b>States' Trend Analysis</b><br>7-Day Rolling average Statistics",
                 margin=dict(t=100,b=0,r=10,l=0), 
                 legend=dict(title='State/Union Territory',font=dict(size=8))) 
fig.show()


# ### Top 5 worst affected states
# In this section, we will take a look at the top 5 worst affected states in India. I have plotted daily cases, deaths and recoveries along with their rolling average go get a sense of the trend: increasing or decreasing.

# In[ ]:


n = 5
df_curr = df_states[df_states['date']==df_states.date.max()]
states = df_curr.nlargest(n,'Confirmed')['state'].values.tolist()


# In[ ]:


def plot_state(state=state):
    '''Geeting the top worst affected states according the avarage value calculated using 7-day rolling'''
    df_state = df_states[df_states['state']==state]
    df_state['roll_avg_c'] = df_state['Daily Cases'].rolling(7).mean()
    df_state['roll_avg_d'] = df_state['Daily Deaths'].rolling(7).mean()
    df_state['roll_avg_r'] = df_state['Daily Cured'].rolling(7).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Cases',x=df_state['date'],y=df_state['Daily Cases'],marker_color='royalblue',yaxis='y'))
    fig.add_trace(go.Scatter(name='7-day rolling avg',x=df_state['date'],y=df_state['roll_avg_c'],marker_color='black',yaxis='y'))
    fig.add_trace(go.Bar(name='Deaths',x=df_state['date'],y=df_state['Daily Deaths'],marker_color='crimson',yaxis='y2'))
    fig.add_trace(go.Scatter(name='7-day rolling avg',x=df_state['date'],y=df_state['roll_avg_d'],marker_color='black',yaxis='y2'))
    fig.add_trace(go.Bar(name='Recoveries',x=df_state['date'],y=df_state['Daily Cured'],marker_color='limegreen',yaxis='y3'))
    fig.add_trace(go.Scatter(name='7-day rolling avg',x=df_state['date'],y=df_state['roll_avg_r'],marker_color='black',yaxis='y3'))
    # Update axes
    fig.update_layout(
        xaxis=dict(
        autorange=True,
        rangeslider=dict(
            autorange=True,
        ),
        type="date"
    ),
    yaxis=dict(
        anchor="x",
        autorange=True,
        domain=[0, 0.3],
        linecolor="royalblue",
        mirror=True,
        showline=True,
        side="right",
        tickfont={"color": "royalblue"},
        tickmode="auto",
        ticks="",
        title='Cases',
        titlefont={"color": "royalblue","size":10},
        type="linear",
        zeroline=False
    ),
    yaxis2=dict(
        anchor="x",
        autorange=True,
        domain=[0.35, 0.65],
        linecolor="crimson",
        mirror=True,
        showline=True,
        side="right",
        tickfont={"color": "crimson"},
        tickmode="auto",
        ticks="",
        title = 'Deaths',
        titlefont={"color": "crimson","size":10},
        type="linear",
        zeroline=False
    ),
    yaxis3=dict(
        anchor="x",
        autorange=True,
        domain=[0.7, 1],
        linecolor="limegreen",
        mirror=True,
        showline=True,
        side="right",
        tickfont={"color": "limegreen"},
        tickmode="auto",
        ticks="",
        title="Recoveries",
        titlefont={"color": "limegreen","size":10},
        type="linear",
        zeroline=False
    )
    )
    fig.update_layout(title=state,showlegend=False,template='plotly_white')
    fig.show()


# In[ ]:


plot_state(states[0])
plot_state(states[1])
plot_state(states[2])
plot_state(states[3])
plot_state(states[5])

