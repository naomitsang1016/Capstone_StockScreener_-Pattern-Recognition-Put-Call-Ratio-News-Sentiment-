# Import yfinance to draw stock data (uncomment if needed)
# pip install yfinance

#for plotting the candlestick chart (uncomment if needede)
#!pip install plotly==5.7.0

#for downloading the stock chart and smoothing out the chart
import yfinance as yf
import datetime
from datetime import date,timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal #for transforming the original data into a smooth curve
import shutil # for moving the charts
import os
import sys
import subprocess

#import talib #for candlestick recognition

from scipy import stats# for calculating the slope of 200-day MA

#for plotting the candlestick chart
import plotly.graph_objects as go


# for recognition
from keras import models #for importing the trained model
import numpy as np #for converting the image into a numpy array
import keras.utils as image #for reading the image
import tensorflow as tf
# Import streamlit
import streamlit as st
from PIL import Image


#for news sentimental analysis
# import nlp module
import nltk
nltk.download('wordnet')
# install vader for sentiment analysis
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# import data processing module
import numpy as np
# import newsapi to collect news headlines and description
from newsapi.newsapi_client import NewsApiClient
from pathlib import Path
import requests
import torch

# check if the library folder already exists, to avoid building everytime you load the pahe
if not os.path.isdir("/tmp/ta-lib"):

    # Download ta-lib to disk
    with open("/tmp/ta-lib-0.4.0-src.tar.gz", "wb") as file:
        response = requests.get(
            "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        )
        file.write(response.content)
    # get our current dir, to configure it back again. Just house keeping
    default_cwd = os.getcwd()
    os.chdir("/tmp")
    # untar
    os.system("tar -zxvf ta-lib-0.4.0-src.tar.gz")
    os.chdir("/tmp/ta-lib")
    os.system("ls -la /app/equity/")
    # build
    os.system("./configure --prefix=/home/appuser")
    os.system("make")
    # install
    os.system("make install")
    # back to the cwd
    os.chdir(default_cwd)
    sys.stdout.flush()

# add the library to our current environment
from ctypes import *

lib = CDLL("/home/appuser/lib/libta_lib.so.0.0.0")
# import library
try:
    import talib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--global-option=build_ext", "--global-option=-L/home/appuser/lib/", "--global-option=-I/home/appuser/include/", "ta-lib"])
finally:
    import talib

#import CNN model
##########################
##########################
#Change .h5 file path here
##########################
##########################
#cloud_model_location='1WvX6uVroSE4DOk6IFEyz3w33kesvs_Fb'
cloud_model_location='https://drive.google.com/u/0/uc?id=1WvX6uVroSE4DOk6IFEyz3w33kesvs_Fb&export=download&confirm=t'
def download_file_from_google_drive(id, destination):
    #URL = "https://docs.google.com/uc?export=download"
    URL = id
    #session = requests.Session()
    response=requests.get(URL, stream = True)
    #response = session.get(URL, params = { 'id' : id }, stream = True)
    #token = get_confirm_token(response)

    #if token:
        #params = { 'id' : id, 'confirm' : token }
        #response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


device = torch.device('cpu')
def load_models():

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/VCP_model_v1.h5")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
    
    model = models.load_model(f_checkpoint)
    #model.eval()
    return model
model = load_models()

########################################################################
####################functions for all ##################################
########################################################################

#################################
#downloading data using yfinance
#################################
def download_data(ticker):
    #get the start date from the end date: given the end date, and we're plotting a 1 year graph, we can calculate the start date
    start_date = datetime.date.today()-datetime.timedelta(days=365)   
    
    
    #download data from yfinance
    return yf.download(ticker, start=start_date, end = datetime.date.today())

#################################
#function for plotting the graph
#################################
def plot_graph(df):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                  open=df['Open'],
                  high=df['High'],
                  low=df['Low'],
                  close=df['Close'])])

    st.plotly_chart(fig, use_container_width=True)

    
########################################################################
####################candlestick pattern function########################
########################################################################

def candlestick_identify(df):
    no_candle_signal = True

    #1) Morning Star(Doji, southern)
    morning_doji_star = talib.CDLMORNINGDOJISTAR(df['Open'], df['High'], df['Low'], df['Close'])
    if morning_doji_star[-1]!=0:
        st.write('Morning Doji star is found on last day.')
        no_candle_signal = False

    #2) Long Line Candle (= long white day)
    # only accept the positive one (bullish)
    long_line_candle = talib.CDLLONGLINE(df['Open'], df['High'], df['Low'], df['Close'])
    if long_line_candle[-1]>0:
        st.write('Long line candle is found on last day.')
        no_candle_signal = False


    #3) Bullish marubozu (= Marubozu, closing white)
    # only accept > 0
    closing_marubozo = talib.CDLCLOSINGMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    if closing_marubozo[-1]>0:
        st.write('Bullish marubozu is found on last day.')
        no_candle_signal = False


    #4) bullish marubozo (= Marubozu, white)
    # only accept value > 0
    marubozo= talib.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
    if closing_marubozo[-1]>0:
        st.write('Bullish marubozu is found on last day.')
        no_candle_signal = False

    if no_candle_signal:
        st.markdown(f'<h3 style="color:#b3aaaa;font-size:24px;">{"Neutral"}</h3>', unsafe_allow_html=True)
        st.write('No candlestick pattern identified.')
    else:
        st.markdown(f'<h3 style="color:#33ff33;font-size:24px;">{"Bullish"}</h3>', unsafe_allow_html=True)



########################################################################
################## functions for consolidation/breakout check ##########
########################################################################

def is_consolidating(df,percentage = 2):
    #we're just interested in the most recent results. So we'll just look at the Adj close of the past 15 days (3 weeks)
    max_price = df[-15:]['Adj Close'].max()
    min_price = df[-15:]['Adj Close'].min()
    if min_price >= (max_price*(1-percentage/100)):
        return True

def is_breaking_out(df):
    #check the Adj close of the past 15 days (excluding today)
    max_price = df[-16:-2]['Adj Close'].max() #-16:-2 --> the last 15 days excluding today
    current_price = df[-1:]['Adj Close'].max() #the .max() is for retreive the price from the series
    if current_price >=  max_price:
        return True


########################################################################
####################functions for template checking ####################
########################################################################

################################################
#finding the slope of the 200-day moving average
################################################
def find_slope(data):
    slope, intercept, r, p, se = stats.linregress(range(len(data)), data)
    return slope


#############################################################
#function for finding the slope of the 200-day moving average
##############################################################
def trend_template_filter(df):
    #calculate the required moving averages (MA) using the closeing prices
    df['50_MA'] = df['Adj Close'].rolling(window=50).mean()
    df['150_MA'] = df['Adj Close'].rolling(window=150).mean()
    df['200_MA'] = df['Adj Close'].rolling(window=200).mean()

    #find out the 52 week high and low
    df['52_week_low']=df["Adj Close"].rolling(window=260).min()
    df['52_week_high']=df["Adj Close"].rolling(window=260).max()

    #1) The current stock price is above both the 150-day and 200-day moving average
    df['Criterion1'] = (df['Close']>df['150_MA']) & (df['Close']>df['200_MA'])

    #2) The 150-day moving average is above the 200-day moving average
    df['Criterion2'] = df['150_MA']>df['200_MA']

    #3) The 200-day moving average is trending up for at least 1 month
    # there are around 20 trading days in a month, therefore window=20 is used
    df['Criterion3'] = df['200_MA'].rolling(window=20).apply(find_slope)>0

    #4) The 50-day moving average is above both the 150-day and 200-day moving average
    df['Criterion4'] = (df['50_MA']>df['150_MA']) & (df['50_MA']>df['200_MA']
                                                    )
    #5) The current stock price is trading above the 50-day moving average
    df['Criterion5'] = df['Adj Close']>df['50_MA']

    #6) The current stock price is at least 30% above it’s 52-week low
    # window=260: 52week * 5 trading days per week 
    df['Criterion6'] = df['Adj Close']>(df['52_week_low']*1.3)

    #7) The current stock price is within at least 25% of it’s 52-week high
    #window=260: 52week * 5 trading days per week 
    df['Criterion7'] = df['Adj Close']>df['52_week_high']*0.75

    #count the numebr of criteria met:
    number_of_criteria_met = df[['Criterion1', 'Criterion2', 'Criterion3',\
                                 'Criterion4','Criterion5','Criterion6','Criterion7']].iloc[-1].sum()

    st.subheader('Mark Minervini\'s Trend Template fulfillment: {}/7'.format(str(number_of_criteria_met)))


    cols = st.columns(3)
    with cols[0]:

        st.write('Is the current stock price above both the 150-day and 200-day moving average?')
        if(df.iloc[-1]['Criterion1']):
            st.write('Yes')
        else:
            st.write('No')

        st.write('Is the 150-day moving average above the 200-day moving average?')
        if(df.iloc[-1]['Criterion2']):
            st.write('Yes')
        else:
            st.write('No')

        st.write('Has the 200-day moving average trending up for at least 1 month?')
        if(df.iloc[-1]['Criterion3']):
            st.write('Yes')
        else:
            st.write('No')

    with cols[1]:
        st.write('Is the 50-day moving average above both the 150-day and 200-day moving average?')
        if(df.iloc[-1]['Criterion4']):
            st.write('Yes')
        else:
            st.write('No')           
    
        
        st.write('Is the current stock price above the 50-day moving average?')
        if(df.iloc[-1]['Criterion5']):
            st.write('Yes')
        else:
            st.write('No')

    with cols[2]:
        st.write('Is the current stock price at least 30% above it\'s 52-week low?')
        if(df.iloc[-1]['Criterion6']):
            st.write('Yes')
        else:
            st.write('No')

        st.write('Is the current stock price within at least 25% of it\'s 52-week high?')
        if(df.iloc[-1]['Criterion7']):
            st.write('Yes')
        else:
            st.write('No')





########################################################################
#################### VCP functions #####################################
########################################################################

###########################################
#function to plot chart for performing CNN
###########################################
def smooth_out_and_save_chart(df):

    #plotting
    x_data = df.index.tolist()      # use date as x-axis
    y_data = 0.5*(df['Low']+df['High']) #use the average of high and low as data
    
    
    

    #filtering the original data with different parameters
    # savgol_filter(input data, window size, polynomial order)
    y_sg_1 = signal.savgol_filter(y_data, 51, 3)
    y_sg_2 = signal.savgol_filter(y_data, 61, 4)
    y_sg_3 = signal.savgol_filter(y_data, 71, 5)
    y_sg_4 = signal.savgol_filter(y_data, 81, 6)

    #open a new graph the plot the filtered chart
    plt.figure(figsize=(10, 6), dpi= 50, facecolor='w', edgecolor='k')


    plt.plot(x_data, y_sg_1)
    plt.plot(x_data, y_sg_2)
    plt.plot(x_data, y_sg_3)
    plt.plot(x_data, y_sg_4)
    

   
    plt.savefig('smooth_chart.png')
    plt.close()



def VCP_recognition(df):

  
  #smooth out the chart and save it as a png file
  smooth_out_and_save_chart(df)

  #read the image
  chart = tf.keras.preprocessing.image.load_img('smooth_chart.png')

  # Convert the image to a numpy array
  chart_array = tf.keras.preprocessing.image.img_to_array(chart)

  #make prediction
  predictions = model.predict(np.expand_dims(chart_array,0))

  if predictions>0.5:
      st.markdown(f'<h3 style="color:#33ff33;font-size:24px;">{"Bullish"}</h3>', unsafe_allow_html=True)
      st.write('VCP pattern observed.')
      
  else:
      st.markdown(f'<h3 style="color:#ff0d00;font-size:24px;">{"Bearish"}</h3>', unsafe_allow_html=True)
      st.write('No VCP pattern observed.')
      
      
  
###Put call ratio

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates - nearerst 3 months
    exps = tk.options
    exps=pd.to_datetime(exps)
    exps=exps[exps<pd.to_datetime(date.today()+ relativedelta(months=1))]
    exps=[i.strftime('%Y-%m-%d') for i in exps]
    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    # Error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options

def contrarian_choice(x): 
    if 0.7>x:
        return 'Strong Bearish'
    elif x>1:
        return 'Strong Bullish'
    elif x==1:
        return 'Neutral'
    else:
        return 'Bearish'
    
def momentum_choice(x): 
    if 0.7>x:
        return 'Strong Bullish'
    elif x>1:
        return 'Strong Bearish'
    elif x==1:
        return 'Neutral'
    else:
        return 'Bullish'  




########################################################################
##########     News sentiment analysis functions    ####################
########################################################################


# Function to get news title, description, sentiment and url
def get_articles_sentiments(NEWS_API_KEY,keywrd, startd, sources_list = None, show_all_articles = False):
    
    # call api
    newsapi = NewsApiClient(api_key = NEWS_API_KEY)
    
    # Make sure start day in datetime format
    if type(startd) == str:
        my_date = datetime.strptime(startd,'%d-%b-%Y')
    else:
        my_date = startd
  
    if sources_list:
        articles = newsapi.get_everything(q = keywrd, #stock name
                                          from_param = my_date.isoformat(), #start date, format e.g. 2022-04-18
                                          to = (my_date + timedelta(days = 1)).isoformat(), #end date, format e.g. 2022-04-18
                                          language="en", #English article
                                          sources = ",".join(sources_list), #define news source if any (try which source more accountable)
                                          sort_by="relevancy", #get the most relevant articles in front
                                          page_size = 100) #only get top 100 pages news
    else:
        articles = newsapi.get_everything(q = keywrd,
                                          from_param = my_date.isoformat(), 
                                          to = (my_date + timedelta(days = 1)).isoformat(),
                                          language="en",
                                          sort_by="relevancy",
                                          page_size = 100)
    
    # Store article titles to check if has been processed or not
    seen = set()
    # Store article title and description for sentiment analysis
    article_content = ''
    # Store details as list to form dataframe
    date_sentiments_list = []
  
    for article in articles['articles']:
        # Check if the the same article has been processed
        if str(article['title']) in seen:
            continue
        else:
            seen.add(str(article['title'])) #store article title for if-clause checking
            article_content = str(article['title']) + '. ' + str(article['description']) #store title and description for sentiment analysis
            sentiment = sia.polarity_scores(article_content)['compound'] #only take the compound value for evaluation
            date_sentiments_list.append((my_date,article['title'],article['description'],sentiment, article['url']))
    
    # Return dataframe with 5 columns sorted by positive as 1st row
    return pd.DataFrame(date_sentiments_list, columns=['Date', 'Title','Description','Sentiment','URL']).sort_values(by='Sentiment', ascending=False).reset_index(drop=True)


# Function to compute sentiment from all and only business news sources and return dataframe with date, sentiments and close price
#Reminder: FREE NewsAPI allows to retrieve only 1 month of news data
#Reminder: Limited to 100 requests over a 24 hour period

# Function to compute sentiment from all and only business news sources and return dataframe with date, sentiments and close price
#Reminder: FREE NewsAPI allows to retrieve only 1 month of news data
#Reminder: Limited to 100 requests over a 24 hour period

def sent_vs_price(NEWS_API_KEY,start_date, end_date, keywrd, stock):
    # Define current day as a variable for while loop use
    current_day = start_date
    # List to store mean sentiment given all news source
    sentiment_all_score = []
    # List to store the dates from start to today
    dates = []
    # String telling the sentiment trend
    sent_trend = ""

    # Starting from the first date
    while current_day <= end_date:
    
        # Store the date into the list
        dates.append(current_day)
    
        # Get news title, description, sentiment and url from all news source
        sentiments_all = get_articles_sentiments(NEWS_API_KEY,keywrd = keywrd, startd = current_day, sources_list = None, show_all_articles= True )
        # Store mean sentiment given all news source
        sentiment_all_score.append(sentiments_all.Sentiment.mean())
    
        # Add one day to current day
        current_day = current_day + timedelta(days=1)
        
    # Create dataframe with date as index and sentiment mean from all and business news sources
    sentiments = pd.DataFrame(list(zip(sentiment_all_score)), index = pd.DatetimeIndex(dates), columns =['All_sources_sentiment'])
    # Set index name as Date
    sentiments.index.name = 'Date'
    # Download stock price data from yfinance and join close price with the above dataframe
    sentiments = sentiments.join(yf.download(stock, start = start_date, end = end_date)['Close']).dropna()
    sentiments.All_sources_sentiment = sentiments.All_sources_sentiment.apply(lambda x: round(x,2))
    
    # Function telling if the sentiment is going positive or negative
    if len(sentiments) > 2:
        day1 = sentiments.All_sources_sentiment[-1]-sentiments.All_sources_sentiment[-2]
        day2 = sentiments.All_sources_sentiment[-2]-sentiments.All_sources_sentiment[-3]
        if day1 < 0 and day2 < 0:
            sent_trend = "News sentiment drops since the day before at " + \
            str(sentiments.All_sources_sentiment[-3]) + " to today " +\
            str(sentiments.All_sources_sentiment[-1])
        elif day1 > 0 and day2 > 0:
            sent_trend = "News sentiment grows since the day before at " + \
            str(sentiments.All_sources_sentiment[-3])  + " to today " + \
            str(sentiments.All_sources_sentiment[-1])
        
        elif day1 < 0 and day2 > 0:
            sent_trend = "News sentiment dropped yesterday from " + \
                str(sentiments.All_sources_sentiment[-3]) + " to " + \
                str(sentiments.All_sources_sentiment[-2]) + " but rebounces today reaching "  + \
                str(sentiments.All_sources_sentiment[-1])
        elif day1 < 0 and day2 > 0:
            sent_trend = "News sentiment grew yesterday from " + str(sentiments.All_sources_sentiment[-3]) + \
                " to " + str(sentiments.All_sources_sentiment[-2]) + " but drops today reaching " +\
                str(sentiments.All_sources_sentiment[-1])
        else:
            sent_trend = "News Sentiment remains stable at " + str(sentiments.All_sources_sentiment[-1])
    elif len(sentiments) == 2:
        day1 = sentiments.All_sources_sentiment[-1]-sentiments.All_sources_sentiment[-2]
        if day1 < 0:
            sent_trend = "News sentiment drops from yesterday " + str(sentiments.All_sources_sentiment[-2]) \
            + "to today" + str(sentiments.All_sources_sentiment[-1])
        elif day1 > 0:
            sent_trend = "News sentiment goes up from yesterday " + str(sentiments.All_sources_sentiment[-2]) \
            + "to today " + str(sentiments.All_sources_sentiment[-1])
        else:
            sent_trend = "News Sentiment remains stable at " + str(sentiments.All_sources_sentiment[-1])
    elif len(sentiments) < 2:
        sent_trend = "Not enough data to analyze news sentiment"
        
    # Return the dataframe for plotting
    return sentiments,sent_trend,sentiments_all


##################################################################################
############################# main function ######################################
##################################################################################


def main():
  ##################################################################################
  ##################################################################################
  ############################# Side Bar User Input ################################
  ##################################################################################
  ##################################################################################

  
  #st.set_page_config(layout="wide")
  
  st.sidebar.title('Stock Screener')
  
  
  st.sidebar.subheader('Which stock to check?')
  stock = st.sidebar.text_input('Ticker Symbol', placeholder = 'e.g. TSLA')
  stock = stock.upper()

  #for defining the consolidating percentage
  consolidating_percentage = st.sidebar.slider('Set the consolidating range (in percentage).', 0, 10, 2)
  

  # button to trigger program run
  st.sidebar.text("") # just for adding white space
  st.sidebar.text("") # just for adding white space

  ##################################################################################
  ############################# Main Page title and warning#########################
  ##################################################################################
  ##################################################################################
  
  ##Only show without input
  
  if stock=='':
    #st.image('https://www.constructconnect.com/hubfs/Blog%20Images%20and%20Media/Stock-Markets-Header-Graphic-New-Jan-05-2021-07-18-56-72-PM.jpg')
    st.title('Stock Screener')
    st.text('We provide: Pattern Recognition - Consolidating/Breakout | Candlstick Pattern | VCP Pattern | Put-Call-Ratio Calculation | News Sentiment Polarization')
    cols=st.columns(2)
    
    pd.set_option('expand_frame_repr', False)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # There are 4 tables on the Wikipedia page
    # we want the last table

    
    # pretty printing of pandas dataframe
    with cols[0]:
        st.subheader('Dow Jones Industrial Average (^DJI)')
        df3 = download_data('^DJI')
        plot_graph(df3)
        st.subheader('NASDAQ Composite (^IXIC)')
        df2 = download_data('^IXIC')
        plot_graph(df2)
    with cols[1]:
        st.subheader('S&P 500 (^GSPC)')
        df1 = download_data('^GSPC')
        plot_graph(df1)
        st.subheader('NASDAQ 100 companies ')
        payload=pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        fifth_table = payload[4]
        nasdaq_100 = fifth_table
        st.dataframe(nasdaq_100)
   
        
        
  # disclaimer at footer, always show
  footer="""
  <style>
  footer{
  visibility: visible;
  }
  footer:after{
    content: 'The content of this program is not an investment advice\
    and does not constitute any offer or solicitation to offer or recommendation of any investment product.';
    display: block;
    position: left;
    color: grey;
    font-size: 12px;
  }
  </style>
  """
  st.markdown(footer, unsafe_allow_html=True)

  #retrieve data and perform detection only after this button is clicked
  if st.sidebar.button('Check it now!'):
    ##################################################################################
    ##################################################################################
    ############################# Main Page Program ##################################
    ##################################################################################
    ##################################################################################
        
    #this is to check whether the stock exists
    ticker = yf.Ticker(stock)
    #info = None #not working anymore as the yfinance 2023-01 update
    #try:
        #check if the stock exists
    #info = ticker.info #not working anymore as the yfinance 2023-01 update

    # if stock exists, run the program
    
    #company_name = ticker.info['longName']#not working anymore as the yfinance 2023-01 update
   

    # stock list csv from https://www.nasdaq.com/market-activity/stocks/screener

    stock_list=pd.read_csv('nasdaq_stocklist_202303.csv')[['Symbol','Name']]
    company_name=stock_list[stock_list['Symbol']==stock]['Name'].tolist()[0]
    st.title('{} {}'.format(stock, company_name))#not working anymore as the yfinance 2023-01 update
    current_price=ticker.history(period='1d')['Close'][0]
    latest_volume=ticker.history(period='1d')['Volume'][0]
    cols=st.columns(2)
    with cols[0]:
        st.subheader('Latest Stock Price: {}'.format(round(current_price,2)))
        #st.subheader('Industry: {}'.format(info['industry'])) #not working anymore as the yfinance 2023-01 update

    with cols[1]:
        st.subheader('Volume: {}'.format(round(latest_volume,2)))
        #st.subheader('P/E: {}'.format(info['trailingPE'])) #not working anymore as the yfinance 2023-01 update
    
    #plot the candlestick chart here, below the volume and price and above the template check
    df = download_data(stock)
    plot_graph(df)
    st.markdown("""---""")
    


    ## VCP Pattern checking
    trend_template_filter(df)

    pattern_cols=st.columns(3)
    with  pattern_cols[0]:
        st.subheader('Consolidating/ Breakout Check')
        if is_consolidating(df,consolidating_percentage):
            st.markdown(f'<h3 style="color:#b3aaaa;font-size:24px;">{"Worth to put it in your watchlist"}</h3>', unsafe_allow_html=True)
            st.write(stock+' is consolidating.')
        else:
            st.markdown(f'<h3 style="color:#b3aaaa;font-size:24px;">{"Neutral"}</h3>', unsafe_allow_html=True)
            st.write(stock + ' is not consolidating.')
        if is_breaking_out(df):
            st.markdown(f'<h3 style="color:#33ff33;font-size:24px;">{"Bullish"}</h3>', unsafe_allow_html=True)
            st.write(stock +' is breaking out.')


    with  pattern_cols[1]:
        st.subheader('Candlstick Pattern Check')
        candlestick_identify(df)

    with  pattern_cols[2]:
        st.subheader('VCP Pattern Check')
        VCP_recognition(df)
        
    ##Put call ratio
    st.markdown("""---""")
    stock_option=options_chain(stock)
    df_Put_Call_Ratio=pd.DataFrame()
    Call_Volume=stock_option[stock_option['CALL']==True].groupby('expirationDate')['volume'].sum()
    Put_Volume=stock_option[stock_option['CALL']==False].groupby('expirationDate')['volume'].sum()
    Put_Call_Ratio_Volume=Put_Volume/Call_Volume
    df_Put_Call_Ratio=pd.concat([df_Put_Call_Ratio,Put_Call_Ratio_Volume,Call_Volume,Put_Volume],axis=1)
    df_Put_Call_Ratio.columns=['PutCallRatio','Call_Volume','Put_Volume']
    df_Put_Call_Ratio['contrarian_investor_choice']=df_Put_Call_Ratio['PutCallRatio'].apply(lambda x:contrarian_choice(x) )
    df_Put_Call_Ratio['momentum_investor_choice']=df_Put_Call_Ratio['PutCallRatio'].apply(lambda x:momentum_choice(x) )
    df_Put_Call_Ratio.reset_index(inplace=True)
    st.subheader('Put-Call-Ratio: {} on {}'.format(round(df_Put_Call_Ratio['PutCallRatio'][0],2), str(df_Put_Call_Ratio['index'][0]).strip('00:00:00')))
    
    cols=st.columns(2)
    with cols[0]:
        st.subheader('Contrarian Investor Choice: ')
        if 'Bearish' in df_Put_Call_Ratio['contrarian_investor_choice'][0]:
            st.markdown(f'<h3 style="color:#ff0d00;font-size:24px;">{"{}".format(df_Put_Call_Ratio["contrarian_investor_choice"][0])}</h3>', unsafe_allow_html=True)
        elif 'Bullish' in df_Put_Call_Ratio['contrarian_investor_choice'][0]:
            st.markdown(f'<h3 style="color:#33ff33;font-size:24px;">{"{}".format(df_Put_Call_Ratio["contrarian_investor_choice"][0])}</h3>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h3 style="color:#b3aaaa;font-size:24px;">{"{}".format(df_Put_Call_Ratio["contrarian_investor_choice"][0])}</h3>', unsafe_allow_html=True)
    with cols[1]:
        st.subheader('Momentum Investor Choice: ')
        if 'Bearish' in df_Put_Call_Ratio['momentum_investor_choice'][0]:
            st.markdown(f'<h3 style="color:#ff0d00;font-size:24px;">{"{}".format(df_Put_Call_Ratio["momentum_investor_choice"][0])}</h3>', unsafe_allow_html=True)
        elif 'Bullish' in df_Put_Call_Ratio['momentum_investor_choice'][0]:
            st.markdown(f'<h3 style="color:#33ff33;font-size:24px;">{"{}".format(df_Put_Call_Ratio["momentum_investor_choice"][0])}</h3>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h3 style="color:#b3aaaa;font-size:24px;">{"{}".format(df_Put_Call_Ratio["momentum_investor_choice"][0])}</h3>', unsafe_allow_html=True)
    with cols[0]:
        st.write('Volume of Option Call: {} USD'.format(int(df_Put_Call_Ratio['Call_Volume'][0])))
    with cols[1]:
        st.write('Volume of Option Put: {} USD'.format(int(df_Put_Call_Ratio['Put_Volume'][0])))
    
    
    ## Financial Data
    ####Not working anymore as yfinance update in 2023-01, ####
    ####the below info will not be provided in the future  ####
    # st.markdown("""---""")
    # st.subheader('Financial Data (in USD)')
    # cols=st.columns(4)
    # with cols[0]:
    #     st.write('Profit Margins: {}'.format(info['profitMargins']))
    #     st.write('Operating Margins: {}'.format(info['operatingMargins']))
    #     st.write('Current Ratio: {}'.format(info['currentRatio']))
    #     st.write('Debt To Equity: {}'.format(info['debtToEquity']))
    # with cols[1]:
    #     st.write('Gross Margins: {}'.format(info['grossMargins']))
    #     st.write('Gross Profits: {}'.format(info['grossProfits']))
    #     st.write('Return On Assets: {}'.format(info['returnOnAssets']))
    #     st.write('Return On Equity: {}'.format(info['returnOnEquity']))
    # with cols[2]:
    #     st.write('Operating Cashflow: {}'.format(info['operatingCashflow']))
    #     st.write('Free Cashflow: {}'.format(info['freeCashflow']))
    #     st.write('Number Of Analyst Opinions: {}'.format(info['numberOfAnalystOpinions']))
    #     st.write('Revenue Per Share: {}'.format(info['revenuePerShare']))
    # with cols[3]:
    #     st.write('Revenue Growth: {}'.format(info['revenueGrowth']))
    #     st.write('Earnings Growth: {}'.format(info['earningsGrowth']))
    #     st.write('Target Mean Price: {}'.format(info['targetMeanPrice']))
    #     st.write('Quick Ratio: {}'.format(info['quickRatio']))
    
   


    #News sentiment analysis
    ###################
    # Define API here #
    ###################

    #Serena's 
    #NEWS_API_KEY = '10c97c512cf34a81b101130fface332d' 
    #NEWS_API_KEY = '264bee2d6a934d66bf3a8e5f6e5f06a8'
    #NEWS_API_KEY = 'bc44334904894bb996cdb01a88d35e0d'

    #Thomas's
    NEWS_API_KEY = '89ae2f1029c9415e886c3f321a29076d' 
    #NEWS_API_KEY = '192773fbfc124e0a8cdda5b4c3093eaf' 

    #Naomi's
    #NEWS_API_KEY = 'ecf7a836f0004036ae5f88f665dcc627'
    #NEWS_API_KEY = '066de2ef1d0446fa9f66beb93898e595'
    #NEWS_API_KEY = '595f99ca87644eb281030b4278b97b0d'

    # Define end date as today
    end_date = date.today()
    # Starting date would be of the day one month prior to today
    start_date = date(year = end_date.year, month = end_date.month, day = end_date.day-5)
    st.markdown("""---""")
    st.subheader('New Sentiment')
    keywrd = company_name
    # keywrd = stock
    tent_vs_price_df,trend,sentiments_all = sent_vs_price(NEWS_API_KEY,start_date, end_date, keywrd, stock)
    
    if ('grows' in trend) or ('grew' in trend):
        st.markdown(f'<h3 style="color:#33ff33;font-size:24px;">{"Bullish"}</h3>', unsafe_allow_html=True)
        st.write(trend)
    elif ('drops' in trend) or ('dropped' in trend):
        st.markdown(f'<h3 style="color:#ff0d00;font-size:24px;">{"Bearish"}</h3>', unsafe_allow_html=True)
        st.write(trend)
    else:
        st.markdown(f'<h3 style="color:#b3aaaa;font-size:24px;">{"Neutral"}</h3>', unsafe_allow_html=True)
        st.write(trend)

    if sentiments_all.shape[0]>0: #if there is any news, print them out
        st.text('Selected News Articles:')
        if sentiments_all.shape[0]>10:
            #select 10 news randomly
            selected_news = sentiments_all.sample(n = 10)

            #print the news title out
            for i in range(10):
                st.write("[{}]({})".format(selected_news.iloc[i]['Title'],selected_news.iloc[i]['URL'] ))
        else: #less than or equal to 10 news: print ALL news
            #print all the news out
            for i in range(sentiments_all.shape[0]):
                st.write("[{}]({})".format(sentiments_all.iloc[i]['Title'],sentiments_all.iloc[i]['URL'] ))
    
    #except:
      #st.write("Beep boop! Cannot get info of ",stock,", it probably does not exist.")
    




#run the program

if __name__ == "__main__":
  main()
