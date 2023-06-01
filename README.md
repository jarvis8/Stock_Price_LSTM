The stock market is a vast place, with many opportunities for people to make it big. However, it is also a very dark place, where people can lose large sums of money if the correct decision is not made. The market changes rapidly, and people can either make a fortune or lose one in a matter of seconds. There are various factors that influence the market, and it is difficult to keep track of each purchased stock to maximise profit. A stock price predictor is a great tool that can help us in making a decision of when to buy, retain, or keep a stock.

As mentioned earlier, there are various factors that affect the pricing of a stock in the market. So we can never predict the future stock price accurately. But what we do know, is that stock prices follow a trend similar to what has previously happened. There are various patterns that often repeat, unless there is some major event that affects the company greatly. 

Given that the future prices are affected by past prices, we should consider using an algorithm or a model that involves previous values while predicting future values. We decided to use the stacked LSTM(Long-Short Term Memory) model. LSTM is a type of recurrent neural network (RNN) that is used to process sequential data, such as time series or natural language text.

In an LSTM model, there are small memory cells that can store information and pass it forward to the next time step. These cells are controlled by "gates" that decide whether to let information in or out. There are three types of gates in an LSTM model:

Forget gate: This gate decides whether to forget the information stored in the memory cell.
Input gate: This gate decides whether to update the memory cell with new information.
Output gate: This gate decides how much information to output from the memory cell.

By using these gates, an LSTM model can selectively store or discard information over long periods of time, making it well-suited for handling long-term dependencies in sequential data.
A stacked LSTM model is an extension of the basic LSTM model that has multiple layers of LSTM cells stacked on top of each other.

In a stacked LSTM model, the output of one LSTM layer is fed as input to the next LSTM layer, with each layer learning a higher-level representation of the input data. The first LSTM layer processes the input sequence and passes the output to the second layer, which processes the output from the first layer, and so on.

The advantage of using a stacked LSTM model is that it can capture more complex relationships in sequential data by learning multiple levels of abstraction. The lower layers can learn simple patterns in the data, while the higher layers can learn more complex patterns that are built on top of the lower-level patterns.

However, it's important to note that stacking too many LSTM layers can lead to overfitting, where the model learns to fit the training data too well and performs poorly on new, unseen data. Therefore, the number of LSTM layers should be chosen based on the complexity of the problem and the amount of training data available.

To summarise, a stacked LSTM model is a type of neural network that consists of multiple layers of LSTM cells stacked on top of each other, which can learn multiple levels of abstraction in sequential data.
