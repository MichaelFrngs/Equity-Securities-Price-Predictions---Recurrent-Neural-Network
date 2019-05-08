#Recurrent Neural Network for predicting stock prices
#Something like "The last 60 minutes of prices" is a sequence. The order counts obviously. This is what a recurrent net is good for.

#To install on pc, type into terminal: pip install pandas-datareader
import pandas_datareader.data as web #This is the pandas data reading API. Lets you read from a website or a source.
import datetime as dt
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing #Allows use of preprocessing.scale() function
from collections import deque #Imports the deque function
import random #allows use of shuffle function
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization #CuDNNLSTM for GPU processing
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard #Model checkpoint allows you to always save just the best epoch instead of ALL of them.

#This is slightly modified from the one created above
def Pull_Daily_Prices(ticker, start_year, start_month, end_year, end_month):
    #Starting date for the desired data
    start = dt.datetime(start_year, start_month, 1)
    #Ending date for the desired data
    end = dt.datetime(end_year, end_month, 1)
    #Pulls daily stock data for the desired ticker and desired dates
    Stock_Data = web.DataReader(ticker, 'iex', start, end).reset_index()   
    #Changes the date column from a string type variable to datetime type. Allows use of dt.methods
    Stock_Data['date'] = pd.to_datetime(Stock_Data['date'])
    #Initiate Variable. We'll fill it with the stock data mid-loop.
    DailyStock_Data = pd.DataFrame()

    #Initiate counters
    i=0
    indexNum = 0
    #Creates monthly data by filtering the daily data.
    while i < Stock_Data.shape[0]: #Counts the number of rows in the stock_data dataframe.
        try:
            print(Stock_Data["date"][indexNum])
            #Creates the dataframe by iteratively appending each row.
            DailyStock_Data = DailyStock_Data.append(pd.DataFrame([Stock_Data.iloc[indexNum,]]),ignore_index=True)
            #Market is open 5 days a week. Aprox 4.34 weeks per month. Result is about 22 market days per month. We only want to select one day a month from the data.
            indexNum = indexNum + 1#math.floor(365.25/7/12*5)  
            i=i+1
        except:
            print("End")
            break
    #Creates dynamically named global variable with end results. Allows for ease of use elsewhere. Format is #Data_TICKER
    globals()["Data_%s" %ticker] = DailyStock_Data
    return(DailyStock_Data)


#Initiate Variable for later. We will append stuff to it.
main_df = pd.DataFrame()

#In this section: A Dirty way of merging a bunch of temporary dataframes
#Names of stocks to pull data from. We will make predictions using this data, but predictions for only one stock (despite using both assets 
#as training data).
StockList = ["AAPL","SPY"]
#iterate through ratios list
for stock in StockList:
    #dynamic path for finding each csv file
    dataset = Pull_Daily_Prices(StockList[StockList.index(stock)],2014,4,2019,4) #Pulls 5 years of data. You can change this as you like.
    time.sleep(2)
    #read each csv file on each iteration
    temporary_df = dataset
    #on every iteration rename stuff. we want to be able to identify where each "close" column came from when everything is finally merged.
    temporary_df.rename(columns={"close": f"{stock}_close", "volume":f"{stock}_volume"}, inplace = True)
    #define the index. It is now the time column. 
    temporary_df.set_index("date", inplace = True)
    #throws away the rest of the data and keeps only the close & volume. We will append this the the permanent dataframe
    temporary_df = temporary_df[[f"{stock}_close",f"{stock}_volume"]]
    #Let's start appending our dynamically formatted data into the permanent dataframe
    #If the main dataframe is empty....
    if len(main_df) == 0:
        main_df = temporary_df
    else:
        #begin joining/appending/merging. W/e you wanna call it. 
        main_df = main_df.join(temporary_df)

#Preview        
print(main_df.head())

#Prints all of the columns
print(main_df.columns)
    
#We just created the sequenetial data
#We need the sequences and now we need some targets. In other words, what is the answer-key?

#How many trailing data points will we use to make predictions?
#If you get this error: ValueError: Error when checking input: expected lstm_3_input to have 3 dimensions, but got array with shape (0, 1), then u must lower seqlen (not enough data)
SEQ_LEN = 30
#Every period is one day. We're predicting 1 day into the future.
FUTURE_PERIOD_PREDICT = 1
STOCK_TO_PREDICT = "AAPL"
EPOCHS = 500
BATCH_SIZE = 64
#You want a name that's descriptive of the model so that you can make a lot of models on tensor board
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

#This function will compute whether we should buy at a given point in time or not.
def classify(current_price,future_price):
    if float(future_price) > float(current_price):
        return 1 #Means buy
    else:
        return 0 #Means don't buy, or short.
    
#scales and normalizes the data
def preprocess_df(df):
    #let's get rid of the future_prices column, since it was only created to create the target column. It's useless now.
    df = df.drop("future_price",1)
    #let's scale the columns by iterating through them
    for col in df.columns:
        #We want to avoid the target column. It's a completed column, so we don't need to do anything to it
        if col != "target":
            #Show what's happening
            print("Raw data:" ,df.head(5))
            #gets percentage return value to replace the price values
            df[col] = df[col].pct_change() # .pct_chng normalizes the data by making it % returns. For example: Berkshire's returns can be compared to MU's returns.
            #show what's happening
            print("Normalized:",df[col])
            #Gets rid of the first value (which is an ERROR) since we can't get a return for that first value
            df.dropna(inplace= True)
            #scales the values of the columns using the preproccessing scale function from skleark.preproccessing package
            df[col] = preprocessing.scale(df[col].values)
            print("Scaled:",df[col])
    #Just in case ;)
    df.dropna(inplace=True)
    #creates empty list
    sequential_data = []    
    #deque's work kind of like a deck of cards. Can only hold a maximum amount of variables. Overwrites the oldest stuff as you continue to append new stuff to it.
    #This deque is going to contain only a the last few days of data that we're going to use to predict the future.
    prev_days = deque(maxlen = SEQ_LEN)
    #Let's print to show what's happening
    print(df.head())
    #The values method converts the dataframe into a list of lists, which means that the time index will be gone. It's still in order though. #I think it's necessary
    #So that we can iterate through the values in each row from left to right.
    for i in df.values: #.values method turns every row into an array. What is ultimately returned is an array of row-arrays. We're iterating through each row.
        #notice this is a list because n for n is between brackets. #n for n in i just returns each element in the i-list.
        #in this case. the i-list is a bunch of arrays (which are the rows), and the elemenets are pretty much the stuff in the row.
        prev_days.append([n for n in i[:-1]]) #-1 index dodges the last column, which is the target column.
        print(prev_days,"END")
        #Check to see if we have enough data in the deque. If so....
        if len(prev_days) == SEQ_LEN:
            #Appends both the features and the label/target/answerkey:i[-1] to an empty list.
            sequential_data.append([np.array(object=prev_days), i[-1]])
    
    random.shuffle(sequential_data)
        
#Let's balance the data ("We won't have a 60:45% buy to sell ratio, only 60:40 or 50:50 etc)
    #empty lists
    buys = []
    sells = []
    
    #Iterates through both columns, first column will be temporarily named seq, and second will be target.
    for seq, target in sequential_data:
        #if a row's target says sell
        if target == 0:
            #append that piece of data to the sell list
            sells.append([seq,target])
        elif target == 1:
            buys.append([seq,target])
    #Likely not necessary. For good measure
    random.shuffle(buys)
    random.shuffle(sells)
    
    #Which list is smaller? Returns the size of the smaller list.
    lower = min(len(buys),len(sells))
    
    #Only selects up to smallest size of the two. This makes sure we have an even number of buys and sells.
    buys = buys[:lower]
    sells = sells[:lower]
    
    #Merge the two lists.
    sequential_data = buys+sells
    #we want to make sure the data is not clumped up as all buys then all sells, cause it'll confuse the computer
    random.shuffle(sequential_data)
    
    #Create empty lists
    X = []
    y = []
    #fill the empty lists from the sequential data object
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    return np.array(X), y
    
#let's create the target column, aka the "answer key". We're just shifting the prices forward for the "answers" since it's 20/20 hindsight.
#We will train on this data. The target is what we're predicting.
main_df["future_price"] = main_df[f"{STOCK_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
#Let's see what the data looks like
print(main_df[[f"{STOCK_TO_PREDICT}_close", "future_price"]].head)

#Create new target column.  map(function, parameter_1_data, parameter_2_data). #Uses the classify function we defined previously.
main_df["target"] = list(map(classify, main_df[f"{STOCK_TO_PREDICT}_close"], main_df["future_price"]))
#let's see a preview of the new column with the other data
print(print(main_df[[f"{STOCK_TO_PREDICT}_close", "future_price","target"]].head(10)))

#We still have to build the sequences, balance the data, normalize, and scale the data.





#what is out-of-sample data? It's data that the model has not been trained on. This is important to test on because otherwise you will overfit.
#We're going to segregate 5% of the data from our main data set so that we can test our model on unseen data later on.

#Makes an array from the index of main_df. Sorted makes sure that the data is in order (in other words, time is flowing in sequence)
times = sorted(main_df.index.values)
#- index gets the last 5% of time stamps. len is the length of the times array.
last_5pct = times[-int(0.05*len(times))]
#Shows how many time stamps make up our out-of-sample data
print(last_5pct) 

#This is our 5% segregated data
validation_main_df = main_df[(main_df.index >= last_5pct)]
#This is the remaining 95% training data. Notice index returns an array of the entire index.
main_df = main_df[(main_df.index < last_5pct)]



#Let's balance the data

#Creates two variables from a function. Scales and normalizes the data
train_x, train_y = preprocess_df(main_df)
#Creates two variables from a function. Scales and normalizes the data
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"length of training data: {len(train_x)} length of validation data: {len(validation_x)}")
#Notice it's balanced.
print(f"Number of Dont buys: {train_y.count(0)} Number of buys: {train_y.count(1)}")
print(f"Validation Don't buys: {validation_y.count(0)} buys: {validation_y.count(1)}")



#Let's Build the neural network

#empty model
model = Sequential()

#Recurrent layer with 128 nodes. Input shapes determines the a0 nodes.
model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences = True))
#Prevents overfitting
model.add(Dropout(0.2))
#Normalizes the output of this layer
model.add(BatchNormalization())


#Recurrent layer #2 with 128 nodes. Input shapes determines the a0 nodes.
model.add(LSTM(128, input_shape = (train_x.shape[1:]), return_sequences = True))
#Prevents overfitting
model.add(Dropout(0.2))
#Normalizes the output of this layer
model.add(BatchNormalization())


#Dense layer comes after this layer (so you remove the return sequences cause dense layers don't understand that) with 128 nodes. Input shapes determines the a0 nodes.
model.add(LSTM(128, input_shape = (train_x.shape[1:])))
#Prevents overfitting
model.add(Dropout(0.2))
#Normalizes the output of this layer
model.add(BatchNormalization())


#Let's add a dense layer
model.add(Dense(32, input_shape = (train_x.shape[1:]), activation = "relu" ))
#Prevents overfitting
model.add(Dropout(0.2))
#Normalizes the output of this layer
model.add(BatchNormalization())


#Let's add the final output layer. Notice we use softmax for output layer
model.add(Dense(32, input_shape = (train_x.shape[1:]), activation = "softmax" ))
#Prevents overfitting
model.add(Dropout(0.2))
#Normalizes the output of this layer
model.add(BatchNormalization())

#Let's pick an optimizer
opt = tf.keras.optimizers.Adam(lr = .001, decay = 1e-6)

#Compile defines the loss function, the optimizer and the metrics. That's all. No training involved.
#read more at: https://stackoverflow.com/questions/47995324/does-model-compile-initialize-all-the-weights-and-biases-in-keras-tensorflow)
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = opt,
              metrics = ["accuracy"])

#Let's make our output reporting system
#Tensorboard is a called a "callback"
#This "board" shows us results and stuff.
#To access tensorboard, open cmd.exe and write: tensorboard --logdir=logs
#If that command gives you problems, downgrade by using: pip install tensorboard==1.12.2   into anaconda promp
#Finally, go to the posted url in the cmd prompt to open tensorboard
tensorboard = TensorBoard(log_dir = f"NeuralNetLogs/{NAME}") #Dynamically created names for our logs. Goes by the model name we defined above
#More tensorboard settings
filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("NeuralNetModels/{}.model".format(
        filepath,monitor = "val_acc",
        verbose = 1,
        save_best_only = True,
        mode = "max"))

#Trains the model. Notice we had to put this AFTER the reporting system because the call back parameters use it.
history = model.fit(
        train_x,
        train_y,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_data = (validation_x,validation_y),
        callbacks = [tensorboard,checkpoint])
