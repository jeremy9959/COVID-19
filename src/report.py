import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from bokeh.plotting import show, output_file, figure
from bokeh.models import ColumnDataSource, LogScale, DatetimeAxis, DatetimeTicker, DaysTicker,Label
from sklearn.linear_model import LinearRegression
output_file(str(datetime.date.today())+'.html')


confirmed_cases=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")



cases = confirmed_cases.groupby('Country/Region').sum().iloc[:,2:].transpose()
cases['total'] = cases.sum(axis=1)

cases.index=pd.to_datetime(cases.index)
cases.index.name='Date'


cases = cases/100
source = ColumnDataSource(cases)
recent_US = np.log(cases['US'][datetime.date.today()-datetime.timedelta(days=8):])
recent_Italy = np.log(cases['Italy'][datetime.date.today()-datetime.timedelta(days=8):])
L = LinearRegression().fit(np.array(range(len(recent_US))).reshape(-1,1),recent_US)
L1 = LinearRegression().fit(np.array(range(len(recent_Italy))).reshape(-1,1),recent_Italy)
percent_rate = 100*(np.exp(L.coef_)-1)[0]
percent_rate_Italy = 100*(np.exp(L1.coef_)-1)[0]
F = figure(width=600,height=600,x_axis_type='datetime',y_axis_type='log',y_range=(.1,2000),x_range=(datetime.datetime(2020,2,1),datetime.datetime(2020,4,1)))


F.line(x='Date',y='US',source=source,line_width=3,color='darkblue',legend_label='US')
F.line(x='Date',y='Italy',source=source,line_width=3,color='green',legend_label='Italy')
F.line(x='Date',y='Germany',source=source, line_width=3,color='green',line_dash='dotted',legend_label='Germany')
F.line(x='Date',y='China',source=source,line_width=3,color='orange',legend_label='China')
#F.line(x='Date',y='predicted',color='red',line_width=3,source=source,line_dash='dotted')
F.line(x='Date',y='total',color='black',line_width=3,line_dash='dashed',source=source,legend_label='World')
F.title.text = "COVID-19 Confirmed Cases (https://github.com/CSSEGISandData/COVID-19)"

F.add_layout(Label(text_font_style='bold',text_font_size="10pt",x=300,y=100,x_units='screen',y_units='screen',text='USDaily Growth Rate: {:.0f} %'.format(percent_rate)))
F.legend.location='bottom_left'
F.legend.background_fill_alpha=0.0
F.add_layout(Label(text_font_style='bold',text_font_size="10pt",x=300,y=85,x_units='screen',y_units='screen',text='Italy Daily Growth Rate: {:.0f} %'.format(percent_rate_Italy)))
F.yaxis.axis_label = "100's of cases (log scale)"
F.xaxis.axis_label="Date"
F.background_fill_color="#EEEEEE"
F.xgrid.grid_line_color="white"
F.ygrid.grid_line_color="white"
show(F)
