import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import spacy

# Download necessary nltk data
nltk.download('stopwords')

# Load spaCy model for POS tagging/NER
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv('unipol_listings_raw.csv')

# Handle missing values
df['weekly_rent'] = pd.to_numeric(df['weekly_rent'], errors='coerce')
df['weekly_rent'] = df['weekly_rent'].fillna(df['weekly_rent'].median())
df['location'] = df['title'].str.extract(r'(\b[A-Z]{1,2}\d{1,2}[A-Z]?\b)', expand=False).fillna('Unknown')
df['description'] = df['description'].fillna('No description provided')

# Drop unneeded fields
df.drop(columns=['bedrooms', 'property_type'], errors='ignore', inplace=True)
df.drop_duplicates(subset='url', keep='first', inplace=True)
df.drop(columns=['listing_id', 'url'], errors='ignore', inplace=True)

# Clean descriptions
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_description'] = df['description'].apply(clean_text)

# POS tagging for amenities
df['amenities'] = df['description'].apply(lambda text: [
    ent.text for ent in nlp(text).ents if ent.label_ in ['FAC', 'ORG', 'PRODUCT']
])
print("Sample amenities:", df['amenities'].head())

# Feature engineering
df['desc_length'] = df['clean_description'].apply(lambda x: len(x.split()))

def classify_unipol_type(desc):
    desc = str(desc).lower()
    if any(x in desc for x in ['homestay', 'host family']):
        return 'Homestay'
    elif any(x in desc for x in ['family', 'dependents']):
        return 'Family Accommodation'
    elif any(x in desc for x in ['en-suite', 'ensuite']):
        return 'En-Suite Room'
    elif 'studio' in desc or 'studio flat' in desc:
        return 'Studio Flat'
    elif 'bedsit' in desc:
        return 'Bedsit'
    elif any(x in desc for x in ['self-contained', 'private flat', 'apartment', '1-bed', '1 bed']):
        return 'Self-Contained Flat'
    elif any(x in desc for x in ['shared', 'shared house', 'shared flat', 'communal', 'house share']):
        return 'Shared House/Flat'
    elif 'room' in desc:
        return 'Room'
    else:
        return 'Unknown'

df['property_type'] = df['description'].apply(classify_unipol_type)
df['location'] = df['location'].astype('category')
df['property_type'] = df['property_type'].astype('category')

# Save
df.to_csv('unipol_listings_cleaned.csv', index=False)




import pandas as pd

# Step 1: Load your cleaned CSV file
listings_df = pd.read_csv('unipol_listings_cleaned.csv')  # make sure this is the correct path

# Step 2: Export only the 'clean_description' column
descriptions = listings_df['clean_description'].dropna()  # Drop NaNs just in case

# Step 3: Define output file path
output_file = 'rental_descriptions_for_sketchengine.txt'

# Step 4: Save each description as a new line in a plain text file
descriptions.to_csv(output_file, index=False, header=False)

print(f"Exported {len(descriptions)} descriptions to {output_file}")

