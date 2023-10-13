import streamlit as st
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Set the correct PIN
correct_pin = "1234"

# Page state
show_pin_prompt = True

# Function to create a word cloud from text
def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def sentiment_label(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Function to calculate term frequency for n-grams
def calculate_term_frequency(text, n):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    # Create n-grams using a sliding window
    ngrams = [tuple(filtered_tokens[i:i+n]) for i in range(len(filtered_tokens) - n + 1)]

    # Count the occurrences of each n-gram
    ngram_freq = Counter(ngrams)

    # Get the top 10 most common n-grams
    top_ngrams = ngram_freq.most_common(10)

    return top_ngrams

def main():
    st.title("Text Analytics App")

    input_mode = ""
    
    # Page state
    show_pin_prompt = True

    # PIN entry
    pin_input_key = "pin_input"
    entered_pin = st.text_input("Enter PIN:", type="password", key=pin_input_key)

    if entered_pin == correct_pin:
        show_pin_prompt = False  # Set pin prompt to False if the entered pin is correct

    # Display content based on PIN entry
    if show_pin_prompt:
        st.error("Incorrect PIN. Please try again.")
    else:
        # Clear the pin prompt section
        st.empty()

        # User input mode: URL, Raw Text, or Review CSV
        st.subheader("Welcome! Choose an input mode:")
        input_mode = st.radio("Select the input mode:", ("Use URL", "Use Raw Text", "Use Review CSV"))

        if input_mode == "Use URL":
            # Rest of the code for handling URL input
            pass
        elif input_mode == "Use Raw Text":
            # Rest of the code for handling raw text input
            pass
        elif input_mode == "Use Review CSV":
            # Rest of the code for handling CSV input
            pass

    if input_mode == "Use URL":
        # User input for URL
        url = st.text_input("Enter URL:")

        # Check if URL is provided
        if url:
            try:
                # Send a GET request to the URL
                response = requests.get(url)
                response.raise_for_status()  # Raise an HTTPError for bad responses

                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract text from all <p> tags
                paragraphs = soup.find_all('p')
                text = '\n'.join([paragraph.get_text() for paragraph in paragraphs])

                # Perform sentiment analysis
                polarity, subjectivity = perform_sentiment_analysis(text)

                # Create word cloud from the extracted text
                wordcloud_figure = create_word_cloud(text)

                # Display the word cloud using st.pyplot()
                st.pyplot(wordcloud_figure)

                # Display sentiment analysis results
                st.subheader("Sentiment Analysis Results:")
                st.write(f"Polarity: {polarity}")
                st.write(f"Subjectivity: {subjectivity}")

                # Short explanations and sentiment label
                st.subheader("Sentiment Analysis Interpretation:")
                st.write("Polarity is a measure of how positive or negative the text is.")
                st.write("Subjectivity is a measure of how subjective or objective the text is.")
                st.write(f"Text Polarity: {sentiment_label(polarity)}")

                # Add a separator between sections
                st.markdown("---")

                # User input for n-gram value
                n_gram = st.number_input("Enter the number of words for n-grams (1 for unigrams, 2 for bigrams, etc.):", min_value=1, max_value=10, value=1)

                # Calculate and display term frequency for n-grams
                st.subheader("Term Frequency Analysis:")
                term_frequency = calculate_term_frequency(text, n_gram)
                st.write(f"Top {n_gram}-grams:")
                for i, (ngram, freq) in enumerate(term_frequency, start=1):
                    st.write(f"{i}. {' '.join(ngram)}: {freq} occurrences")

            except requests.exceptions.RequestException as e:
                st.error(f"Error: {e}")
        else:
            st.info("Enter a URL above to generate a word cloud.")

    elif input_mode == "Use Raw Text":
        raw_text = st.text_area("Enter text:")

        if raw_text:
            # Perform sentiment analysis
            polarity, subjectivity = perform_sentiment_analysis(raw_text)

            # Create word cloud from the entered text
            wordcloud_figure = create_word_cloud(raw_text)

            # Display the word cloud using st.pyplot()
            st.pyplot(wordcloud_figure)

            # Display sentiment analysis results
            st.subheader("Sentiment Analysis Results:")
            st.write(f"Polarity: {polarity}")
            st.write(f"Subjectivity: {subjectivity}")

            # Short explanations and sentiment label
            st.subheader("Sentiment Analysis Interpretation:")
            st.write("Polarity is a measure of how positive or negative the text is.")
            st.write("Subjectivity is a measure of how subjective or objective the text is.")
            st.write(f"Text Polarity: {sentiment_label(polarity)}")

            # Add a separator between sections
            st.markdown("---")

            # User input for n-gram value
            n_gram = st.number_input("Enter the number of words for n-grams (1 for unigrams, 2 for bigrams, etc.):", min_value=1, max_value=10, value=1)

            # Calculate and display term frequency for n-grams
            st.subheader("Term Frequency Analysis:")
            term_frequency = calculate_term_frequency(raw_text, n_gram)
            st.write(f"Top {n_gram}-grams:")
            for i, (ngram, freq) in enumerate(term_frequency, start=1):
                st.write(f"{i}. {' '.join(ngram)}: {freq} occurrences") 

        else:
            st.info("Enter some text above to generate a word cloud.")

    elif input_mode == "Use Review CSV":
        csv_file = st.file_uploader("Upload a CSV file containing reviews:", type=["csv"])

        if csv_file is not None:
            # Load CSV data into a DataFrame
            df = pd.read_csv(csv_file)

            # Concatenate all reviews into a single text
            text = "\n".join(df['Review'])

            # Perform sentiment analysis
            polarity, subjectivity = perform_sentiment_analysis(text)

            # Create word cloud from the concatenated reviews
            wordcloud_figure = create_word_cloud(text)

            # Display the word cloud using st.pyplot()
            st.pyplot(wordcloud_figure)

            # Display sentiment analysis results
            st.subheader("Sentiment Analysis Results:")
            st.write(f"Polarity: {polarity}")
            st.write(f"Subjectivity: {subjectivity}")

            # Short explanations and sentiment label
            st.subheader("Sentiment Analysis Interpretation:")
            st.write("Polarity is a measure of how positive or negative the text is.")
            st.write("Subjectivity is a measure of how subjective or objective the text is.")
            st.write(f"Text Polarity: {sentiment_label(polarity)}")

        else:
            st.info("Upload a CSV file above to generate a word cloud from reviews.")

# Run the Streamlit app
if __name__ == '__main__':
    main()

