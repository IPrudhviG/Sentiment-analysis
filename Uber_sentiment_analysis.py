#import dependencies, download VADER store it to wd
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon',download_dir='/Users/prudhvileo/leoenv/bin/.venv/nltk_data')

#Read the csv using pandas from downloads
df = pd.read_csv("/Users/prudhvileo/Downloads/uber_customer_reviews.csv")
df.head()

# Create a variable sia
sia = SentimentIntensityAnalyzer()

#Calculate polarity scores making sure all entries are strings from content column
df['sentiment'] = df['content'].apply(lambda x : sia.polarity_scores(str(x))['compound'])
df['sentiment_label'] = df['sentiment'].apply(lambda x : 'positive' if x>0 else 'negative' if x<0 else 'neutral')

print(df[['content','sentiment','sentiment_label']])

#Calculate correlation
correlation = df[['sentiment','score']].corr()
print("Correlation between sentiment and score: ")
print(correlation)


#Create a csv of dataset including sentiment analysis values
df.to_csv('sentiment_analysis_results.csv', index=False)


