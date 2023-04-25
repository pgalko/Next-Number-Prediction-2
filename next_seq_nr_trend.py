
'''This script predict the trend of a time series dataset using OpenAI's GPT-3.5-turbo model.
 It preprocesses data, converts it into sequences, and interacts with the AI model through a chat-based prompt. 
 It iterates through the dataset, evaluates the model's performance, calculates prediction error and tokens used, and plots the accuracy.
'''

import pandas as pd
import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt

from array_similarity_search import similarity_search

#-------------------PREPARE DATA-------------------#

# Set the number of values in the sequence to predict the next value.
SEQUENCE_SIZE = 5

# Set the number of predictions to get. Make sure you allow for enough training data for the model. "80% train/20 test" split is recommended.
MAX_ITERATIONS = 800

# Set the number of the most similar question:answer pairs
MAX_EXAMPLES = 3

# Set the name of the column containing the data to be predicted.
data_column = 'Data'

# Set the name of the column containing the dates.
date_column = 'Date'

# Read the data from the CSV file.
data = pd.read_csv("Timeseries_DJI.csv")

# Drop all rows that have NaN values in the predicted column.
data = data.dropna(subset=[data_column])

# Keep only the date and data columns.
data = data[[date_column,data_column]]

# Convert the date column to a datetime object.
data[date_column]= pd.to_datetime(data[date_column])

# Sort the data by date and reset the index.
data = data.sort_values(date_column, ascending=True)
data = data.reset_index(drop=True)

# Convert the data column to a list.
data_list = data[data_column].tolist()

# Remove the elements from the beginning of the list to make the length a multiple of SEQUENCE_SIZE
data_list = data_list[len(data_list) % SEQUENCE_SIZE:]

# Define a function to convert the data into sequences.
def to_sequences(seq_size, obs, MAX_ITERATIONS):
    x, y, z = [], [], []
    for i in range(len(obs)-seq_size):
        # Create a window of data of length seq_size.
        window = obs[i:(i+seq_size)]
        
        # Get the value after the window.
        after_window = obs[i+seq_size]
        
        # Append the after_window value to the z list.
        z.append(after_window)
        
        # Get the last value in the window to be used later as in a final prediction.
        last_after_window = after_window

        # Subtract the after window value from the last window value and assign to after_window variable.(Difference between the predicted value and the previous value)
        # This is to make the data stationary, otherwise the model will be biased towards the last value in the window.
        after_window_trend = after_window - window[-1]

        # If after window_trend is negative after_window = -1, if after_window_trend is positive after_window = 1, if after_window_trend is 0 after_window = 0.
        if after_window_trend < 0:
            after_window = -1
        elif after_window_trend > 0:
            after_window = 1    
        else:
            after_window = 0
        
        # Add the window and after_window to the x and y lists.
        x.append(window)
        y.append(after_window)

    # Get the the last MAX_ITERATIONS values from the list.
    sequences = x[-MAX_ITERATIONS:]
    predictions = y[-MAX_ITERATIONS:]
    actual_values = z[-MAX_ITERATIONS:]

    # Get the values from MAX_ITERATIONS to the beginning of the list.
    train_sequences = x[:-MAX_ITERATIONS]
    train_predictions = y[:-MAX_ITERATIONS]

    # Return x and y numpy arrays and the last value from the y array.
    return np.array(sequences),np.array(predictions),int(last_after_window),np.array(train_sequences),np.array(train_predictions),np.array(actual_values)

# Use the to_sequences function to create sequences and predictions.
sequences, predictions,last_prediction,train_sequences,train_predictions,actual_values = to_sequences(SEQUENCE_SIZE, data_list,MAX_ITERATIONS)
# Convert to integers
sequences = sequences.astype(int)
predictions = predictions.astype(int)
train_sequences = train_sequences.astype(int)
train_predictions = train_predictions.astype(int)
actual_values = actual_values.astype(int)

# Print each row of the sequences and predictions
for i in range(len(sequences)):
    print("{} - {}".format(sequences[i],predictions[i]))

#-------------------CREATE TEMPLATE-------------------#

examples = []

# Define the template.
task = """You are an AI analyst and your task is to predict the next number in a sequence of numbers.
What is the next number following this sequence:{} ?
If the number is lower than the last number in the sequence, the response should be -1.
If the number is higher than the last number in the sequence, the response should be 1.
If the number is the same as the last number in the sequence, the response should be 0.
The response must be a single number without context.
As an example, here are some similar Sequences => Responses:{}.'''
"""

#-------------------CREATE MODEL-------------------#

import openai

# Openai_API_key
api_key = os.environ.get('OPENAI_API_KEY')
# Configure OpenAI and Pinecone
openai.api_key = api_key

# Function to call the OpenAI API
def llm_call(
    messages: str,
    model: str = 'gpt-3.5-turbo',
    temperature: float = 0,
    max_tokens: int = 2,
    ):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    return content, tokens_used

#-------------------RUN MODEL-------------------#

total_tokens_used = []
# Initialize the prompt with the first sequence.
prompt = task.format(sequences[0],examples)
# Initialize the messages list with the prompt.
messages = [{"role": "user", "content": prompt}]

iterations = 0

actual_list = []
predicted_list = []
correct_answers = []
incorrect_answers = []

# Main loop to run the model, and modify the prompt based on the model's response.
for i in range(0,MAX_ITERATIONS):
    print("Iteration:{}".format(i+1))

    # Call the OpenAI API.
    try:
        next_nr,tokens_used = llm_call(messages)
    except openai.error.RateLimitError:
        print(
            "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
        )
        time.sleep(10)  # Wait 10 seconds and try again
        next_nr,tokens_used = llm_call(messages)

    # Append tokens_used to total_tokens_used and sum the total tokens used.
    total_tokens_used.append(tokens_used)

    # Sanitize the LLM response.Strip any non 0 or 1 or -1 characters and spaces from the response.
    next_nr = re.findall(r'(-1|0|1)', next_nr)
    next_nr = ''.join(next_nr)
    
    # Call the similarity search function to get the examples from the unseen portion of the dataset that are most similar to the predicted sequence.
    sorted_indices, sorted_target_arrays,sorted_target_results,sorted_errors = similarity_search(sequences[i],train_sequences,train_predictions,MAX_EXAMPLES)
    
    # Iterate through the sorted indices and replace the values in the examples list with the new values.
    examples = []
    for j in range(len(sorted_indices)):
        examples.append(str(sorted_target_arrays[j]) + ' => ' + str(sorted_target_results[j]))
    
    # reverse the examples list so the most similar example is last in a list of examples
    examples = examples[::-1]

    print("Examples:{}".format(examples))
    
    # if first iteration append responce to messages list.
    if i == 0:
        messages.append({"role": "assistant", "content": next_nr})
    else:
        # Replace the second element in the messages list with the model's response.
        messages[1] = {"role": "assistant", "content": next_nr}

    # If this is the last iteration, modify the sequence that will be used for prediction.
    # This is necessary if predicting on the whole timeseries dataset.
    if i == MAX_ITERATIONS-1:
        # Remove the first element from the sequences[i] and replace the last element with the predictions[i].
        seq4pred = np.delete(sequences[i], 0)
        seq4pred = np.append(seq4pred, last_prediction)
    else:
        seq4pred = sequences[i+1]

    # Compare the next_nr to the actual next number in the sequence. If the prediction is correct change the task template.
    if next_nr == str(predictions[i]):
        print("Actual:{}-Predicted:\033[92m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = '''Correct! Congratulations! Now try with a new sequence:{}.
        If the number is lower than the last number in the sequence, the response should be -1.
        If the number is higher than the last number in the sequence, the response should be 1.
        If the number is the same as the last number in the sequence, the response should be 0.
        The response must be a single number without any context.
        As an example, here are some similar Sequences => Responses:{}.'''
        # Append the sequence and the answer to the correct answers list.
        correct_answers.append([sequences[i],next_nr])
        prompt = task.format(seq4pred,examples)
    else:
        print("Actual:{}-Predicted:\033[91m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = '''Incorrect. The number that follows {} is {}. Now try with a new sequence:{}. 
        If the number is lower than the last number in the sequence, the response should be -1.
        If the number is higher than the last number in the sequence, the response should be 1.
        If the number is the same as the last number in the sequence, the response should be 0.
        The response must be a single number without any context.
        As an example, here are some similar Sequences => Responses:{}.'''
        # Append the sequence and the answer to the correct answers list.
        incorrect_answers.append([sequences[i],next_nr])
        prompt = task.format(sequences[i],predictions[i],seq4pred,examples)

    # If first iteration append prompt to messages list.
    if i == 0:
        messages.append({"role": "user", "content": prompt})
        new_init_prompt = '''You are an AI analyst and your task is to predict the next number in a sequence of numbers.
        If the predicted number is lower than the previous number, the response should be -1.
        If the predicted number is higher than the previous number, the response should be 1.
        If the predicted number is the same as the previous number, the response should be 0.
        The response must be a single number without any context.
        '''
        messages[0] = {"role": "user", "content": new_init_prompt}
    else:
        # Replace the last element in the messages list with the prompt.
        messages[2] = {"role": "user", "content": prompt}

    # Append the actual and predicted values to the lists.
    actual_list.append(predictions[i])
    predicted_list.append(int(next_nr))

    # If this is the last iteration, run the model for one more time to get the final prediction.
    if i == MAX_ITERATIONS-1:
        messages = messages
        next_nr,tokens_used = llm_call(messages)
        total_tokens_used.append(tokens_used)
        print("\033[94mFinal Prediction: {} => {}\033[0m".format(seq4pred,next_nr))

    iterations += 1

#-------------------EVALUATE LLM, AND GET REASONING BEHIND ANSWERS-------------------#

actual_array = np.array(actual_list)
predicted_array = np.array(predicted_list)
# Next_nr cumsum to asess randomness of predictions.
predicted_cumsum_array = np.cumsum(np.array(predicted_list))

# Create a new result array. If the number in actual array differs from the number in predicted array, the result is 1. Otherwise it is 0.
accuracy_array = np.where(actual_array != predicted_array, 1, 0)
# Calcutate the percentage of incorrect predictions.
result_percentage = np.sum(accuracy_array)/len(accuracy_array)*100

print("\n\033[95mGPT-3.5-turbo Prediction Error: {}%\033[0m".format(result_percentage))
# Print the number of correct answers and incorrect answers.
print("Number of correct answers: {}".format(len(correct_answers)))
print("Number of incorrect answers: {}".format(len(incorrect_answers)))

str_correct_answers = []
for i in range(0,len(correct_answers)):
    str_correct_answers.append("{} => {}".format(correct_answers[i][0],correct_answers[i][1]))
str_incorrect_answers = []
for i in range(0,len(incorrect_answers)):
    str_incorrect_answers.append("{} => {}".format(incorrect_answers[i][0],incorrect_answers[i][1]))

# Prompt the LLM to explain the reasoning behind the correct answers.

task2 = '''As an AI analyst you have been given the following assingment "{}" These are your correct answers: {},
and these are the ones that you answered incorrectly: {}.
Can you please closely inspect the number sequences leading to your correct and incorrect answers,
and offer a way how to better analyse the sequences and improve your performance in the next iteration?
Keep your answer as consise as possible".
'''

prompt = task2.format(task,correct_answers[:20],incorrect_answers[:20])
messages = [{"role": "user", "content": prompt}]
reasonning = llm_call(messages,max_tokens=1000,temperature=0.7)
print("\n\033[94mReasoning behind the prediction: \n{}\033[0m".format(reasonning[0]))
print("Tokens used: {}".format(reasonning[1]))

#------------------TOTAL TOKENS USAGE------------------#

# Print the total tokens used.
total_tokens_used.append(reasonning[1])
print("\n\033[93mTotal tokens used: {}\033[0m".format(sum(total_tokens_used)))
print("\033[93mCost(USD): ${}\033[0m".format((sum(total_tokens_used)/1000)*0.002))

#-------------------PLOT THE RESULTS-------------------#

import matplotlib.pyplot as plt

# Create a new array with a running total. If the value in accuracy array is one increment. If it is 0, decrement.
sum_array = np.zeros(len(accuracy_array))
for i in range(1,len(accuracy_array)):
    if accuracy_array[i] == 1:
        sum_array[i] = sum_array[i-1]+1
    else:
        sum_array[i] = sum_array[i-1]-1

fig, ax1 = plt.subplots(figsize=(20, 10))

# Plot error running total on the first y-axis
ax1.bar(range(len(sum_array)), sum_array, alpha=0.5)
z = np.polyfit(range(len(sum_array)), sum_array, 2)
p = np.poly1d(z)
ax1.plot(range(len(sum_array)), p(range(len(sum_array))), "r--")
ax1.set_title("Error Running Total & Actual Values")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Cumulative Error")
# Legend
ax1.legend(["Poly Cumulative Error (Best fit)", "Cumulative Prediction Error"], loc="lower left")

# Plot actual_values on the second y-axis
ax2 = ax1.twinx()
ax2.plot(range(len(actual_values)), actual_values, "b--")
ax2.set_ylabel("Values")
# Legend
ax2.legend(["Actual Values (Original)"], loc="lower right")

plt.show()

#-------------------SAVE THE RESULTS-------------------#

# Create a new dataframe with the results.
df = pd.DataFrame({"Actual_Trend":actual_array,"Predicted_Trend":predicted_array,"Values":actual_values})    
# Save the dataframe to a csv file.
df.to_csv("results.csv",index=False)    
