import pandas as pd
import numpy as np
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import Counter

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- LOAD AND CLEAN DATA ---
df = pd.read_csv('unipol_listings_cleaned.csv')
df['deposit_cleaned'] = df['deposit'].str.replace(r'[^\d.]', '', regex=True)
df['deposit_cleaned'] = pd.to_numeric(df['deposit_cleaned'], errors='coerce')
df = df.dropna(subset=['location', 'property_type', 'rent_includes', 'description', 'weekly_rent', 'deposit_cleaned'])

# --- ADD SKETCH ENGINE FEATURES ---
sketch_phrases = {
    # From 2-grams
    'fully furnished': 'mentions_fully_furnished',
    'double bedrooms': 'mentions_double_bedrooms',
    'fitted kitchen': 'mentions_fitted_kitchen',
    'bills package': 'mentions_bills_package',
    'train station': 'mentions_near_train_station',
    'broadband ultrafast': 'mentions_broadband_ultrafast',
    'internet tv': 'mentions_internet_tv',
    'modern kitchen': 'mentions_modern_kitchen',
    'city centre': 'mentions_city_centre',
    'close university': 'mentions_close_university',
    'en suite': 'mentions_en_suite',
    'newly refurbished': 'mentions_newly_refurbished',
    'spacious living': 'mentions_spacious_living',
    'ideal students': 'mentions_ideal_for_students',
    'student living': 'mentions_student_living',
    'luxury apartment': 'mentions_luxury_apartment',
    'short walk': 'mentions_short_walk',
    'quiet street': 'mentions_quiet_street',
    'prime location': 'mentions_prime_location',
    'fully fitted': 'mentions_fully_fitted',

    # From selected 3-grams
    'hyde park picture': 'mentions_hyde_park_area',
    'royal park pub': 'mentions_royal_park_pub',
    'burley park train': 'mentions_burley_park_station',
    'park train station': 'mentions_park_train_station',
    'distance university leeds': 'mentions_distance_university_leeds',
    'leeds beckett headingley': 'mentions_leeds_beckett_headingley',
    'leeds arts university': 'mentions_leeds_arts_university',
    'leeds trinity university': 'mentions_leeds_trinity_university'
}

for phrase, feature_name in sketch_phrases.items():
    df[feature_name] = df['description'].apply(
        lambda x: 1 if pd.notnull(x) and phrase in x.lower() else 0
    )

# --- CLEAN AMENITIES ---
amenity_keywords = [
    # From curated 1-grams
    'internet', 'wifi', 'gym', 'garden', 'laundry', 'dishwasher', 'fridge', 'oven', 'balcony', 'parking',

    # From 2-grams
    'tv licence', 'internet tv', 'heating gas', 'water supply', 'fitted kitchen',
    'train station', 'double bedrooms', 'bills package', 'transport links',
    'broadband ultrafast', 'council tax', 'private bathroom', 'laundry facilities',
    'washing machine', 'gym access', 'communal garden', 'secure entry',
    'bike storage', 'dishwasher', 'fridge freezer', 'oven hob', 'microwave oven',

    # From selected 3-grams
    'internet tv licence', 'electric supply mains', 'council tax band', 'gas elec water'
]

negation_patterns = [
    r'no\s+{}', r'not\s+{}', r'without\s+{}', r'{}[^a-z]*not included', r'{}[^a-z]*excluded', r'no access to\s+{}'
]

def extract_clean_amenities(text):
    if pd.isna(text):
        return []
    text = text.lower()
    found = set()
    for kw in amenity_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text):
            if not any(re.search(p.format(re.escape(kw)), text) for p in negation_patterns):
                found.add(kw)
    return list(found)

df['clean_amenities'] = df['description'].apply(extract_clean_amenities)

mlb = MultiLabelBinarizer()
amenities_matrix = pd.DataFrame(mlb.fit_transform(df['clean_amenities']),
                                columns=[f"amenity_{a.replace(' ', '_')}" for a in mlb.classes_])
amenity_counts = amenities_matrix.sum()
valid_amenities = amenity_counts[amenity_counts >= 10].index.tolist()
filtered_amenities = amenities_matrix[valid_amenities]
df = pd.concat([df, filtered_amenities], axis=1)
top_amenity_features = list(filtered_amenities.columns)

# --- WORD2VEC CLUSTER FEATURES ---
def generate_word_clusters(descriptions, num_clusters=5):
    sentences = descriptions.dropna().apply(str.split).tolist()
    w2v_model = Word2Vec(sentences, vector_size=50, window=5, min_count=2, workers=4)
    word_vectors = np.array([w2v_model.wv[word] for word in w2v_model.wv.index_to_key])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(word_vectors)
    word_cluster_map = dict(zip(w2v_model.wv.index_to_key, kmeans.labels_))
    return word_cluster_map, num_clusters

word_cluster_map, num_clusters = generate_word_clusters(df['description'])

cluster_flags = [f'cluster_{i}_flag' for i in range(num_clusters)]
for i in range(num_clusters):
    df[cluster_flags[i]] = df['description'].apply(lambda x: any(word_cluster_map.get(w, -1) == i for w in str(x).split()))

# --- SAVE CLUSTER TOP WORDS TO FILE ---
def clean_cluster_words(word_list):
    return [w for w in word_list if w.lower() not in stop_words and w.isalpha() and len(w) > 2]

cluster_words = {i: [] for i in range(num_clusters)}
for word, cluster_id in word_cluster_map.items():
    cluster_words[cluster_id].append(word)

export_data = []
for cid, words in cluster_words.items():
    cleaned = clean_cluster_words(words)
    for w in cleaned[:30]:  # take top 30 words for each cluster
        export_data.append({'cluster': cid, 'word': w})

pd.DataFrame(export_data).to_csv('cluster_top_words_cleaned.csv', index=False)

# --- FINAL DATA PREP ---
sketch_feature_cols = list(sketch_phrases.values())
model_cols = ['location', 'property_type', 'rent_includes', 'description'] + top_amenity_features + cluster_flags + sketch_feature_cols

df_model = df.dropna(subset=model_cols + ['weekly_rent', 'deposit_cleaned'])

X = df_model[model_cols]
y_rent = df_model['weekly_rent']
y_deposit = df_model['deposit_cleaned']

preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(max_features=100), 'description'),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['location', 'property_type', 'rent_includes'])
], remainder='passthrough')


# --- MODEL TRAINING ---
# Weekly Rent
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_rent, test_size=0.2, random_state=42)
rent_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rent_model.fit(X_train_r, y_train_r)
y_pred_rent = rent_model.predict(X_test_r)
y_pred_all = rent_model.predict(X)

# Deposit
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_deposit, test_size=0.2, random_state=42)
deposit_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
deposit_model.fit(X_train_d, y_train_d)
y_pred_deposit = deposit_model.predict(X_test_d)

# --- EVALUATION ---
print("\nWeekly Rent Prediction")
print("MSE:", mean_squared_error(y_test_r, y_pred_rent))
print("MAE:", mean_absolute_error(y_test_r, y_pred_rent))
print("R²:", r2_score(y_test_r, y_pred_rent))

print("\nDeposit Prediction")
print("MSE:", mean_squared_error(y_test_d, y_pred_deposit))
print("MAE:", mean_absolute_error(y_test_d, y_pred_deposit))
print("R²:", r2_score(y_test_d, y_pred_deposit))

# --- ANOMALY DETECTION ---
df_errors = pd.DataFrame({
    'actual': y_rent,
    'predicted': y_pred_all,
    'error': np.abs(y_rent - y_pred_all)
})
threshold = df_errors['error'].quantile(0.95)
anomalies = df_errors[df_errors['error'] >= threshold]

anomalies_detailed = df_model.loc[anomalies.index, ['title', 'location', 'property_type']].copy()
anomalies_detailed[['actual', 'predicted', 'error']] = anomalies[['actual', 'predicted', 'error']].values
anomalies_detailed.to_csv("rent_anomalies_detailed.csv", index=False)

# --- GRAPHS ---
# Figure 1: Rent distribution
plt.figure(figsize=(8,6))
sns.histplot(df['weekly_rent'], bins=20, kde=True)
plt.title('Distribution of Weekly Rent')
plt.xlabel('Weekly Rent (£)')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.savefig('figure1_rent_distribution.png')
plt.show()

# Figure 2: Boxplot by property type
plt.figure(figsize=(10,6))
sns.boxplot(x='property_type', y='weekly_rent', data=df, palette='pastel', showfliers=False)
plt.xticks(rotation=30, ha='right')
plt.title('Weekly Rent by Property Type (Outliers Hidden)')
plt.tight_layout()
plt.savefig('figure2_rent_property_type.png')
plt.show()

# Figure 3: Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_rent, y_pred_all, alpha=0.5)
plt.plot([y_rent.min(), y_rent.max()], [y_rent.min(), y_rent.max()], 'r--')
plt.xlabel('Actual Weekly Rent (£)')
plt.ylabel('Predicted Weekly Rent (£)')
plt.title('Predicted vs Actual Weekly Rent')
plt.grid(True)
plt.tight_layout()
plt.savefig('figure3_predicted_vs_actual.png')
plt.show()

# Figure 4: Prediction error distribution
plt.figure(figsize=(10,6))
plt.hist(df_errors['error'], bins=30, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2)
plt.title('Prediction Error Distribution')
plt.xlabel('Absolute Error (£)')
plt.ylabel('Number of Listings')
plt.tight_layout()
plt.savefig('figure4_error_distribution.png')
plt.show()

# Figure 5: Top 10 anomalies
top_anomalies = anomalies_detailed.sort_values('error', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x='error', y='title', data=top_anomalies, palette='viridis')
plt.title('Top 10 Rental Anomalies by Prediction Error')
plt.xlabel('Absolute Error (£)')
plt.ylabel('Listing Title')
plt.tight_layout()
plt.savefig('figure5_top_anomalies.png')
plt.show()

# --- CLUSTER TOP WORDS PLOTS ---
clusters = pd.read_csv('cluster_top_words_cleaned.csv')

# Combine all descriptions and count word frequencies
all_words = ' '.join(df['description'].dropna()).split()
word_freq = Counter([w.lower() for w in all_words if w.isalpha()])

# Loop through clusters
for cluster_id in clusters['cluster'].unique():
    cluster_words = clusters[clusters['cluster'] == cluster_id]['word']
    frequencies = [word_freq.get(word.lower(), 0) for word in cluster_words]

    height = max(4, 0.5 * len(cluster_words))  # scale height
    plt.figure(figsize=(10, height))
    bars = plt.barh(cluster_words, frequencies, color='skyblue', edgecolor='black')
    plt.gca().invert_yaxis()

    # Add frequency values to bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 str(frequencies[i]), va='center', fontsize=9)

    plt.title(f'Cluster {cluster_id}: Top Words by Frequency in Descriptions', fontsize=12, pad=20)
    plt.xlabel('Frequency', fontsize=11)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=10)

    # Ensure layout fits and prevent title clipping
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # add margin above top of plot

    plt.savefig(f'figure_cluster_{cluster_id}_frequency.png')
    plt.show()


# --- RENTAL FAIRNESS CHECK FUNCTION ---
def check_anomaly(input_data):
    """
    input_data: dict with keys matching model_cols, plus 'weekly_rent' (actual observed)
    """
    input_df = pd.DataFrame([input_data])
    input_df['description'] = input_df['description'].fillna("No description")

    # Generate Sketch features
    for phrase, feature_name in sketch_phrases.items():
        input_df[feature_name] = input_df['description'].str.lower().apply(lambda x: int(phrase in x))

    # Generate amenity features
    for col in top_amenity_features:
        phrase = col.replace("amenity_", "").replace("_", " ")
        input_df[col] = input_df['description'].str.lower().apply(lambda d: int(phrase in d))

    # Generate cluster flags
    for i in range(num_clusters):
        input_df[f'cluster_{i}_flag'] = input_df['description'].apply(
            lambda x: any(word_cluster_map.get(w, -1) == i for w in str(x).split())
        )

    # Ensure all required model columns are present
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0  # default for missing binary/flag features

    # Subset for prediction
    X_input = input_df[model_cols]

    # Predict and compare
    predicted_rent = rent_model.predict(X_input)[0]
    actual_rent = input_data['weekly_rent']
    error = abs(actual_rent - predicted_rent)

    print("\n--- Fairness Check Result ---")
    print(f"Predicted Rent: £{predicted_rent:.2f}")
    print(f"Actual Rent:    £{actual_rent:.2f}")
    print(f"Prediction Error: £{error:.2f}")
    print("Result: " + ("ANOMALY DETECTED" if error >= threshold else "Normal"))

    return error >= threshold

# --- USER PROMPT FOR RENTAL CHECK ---
try:
    print("\n--- Rental Fairness Manual Checker ---")
    loc = input("Enter location (postcode like LS6): ")
    ptype = input("Enter property type (e.g., Room, Studio Flat): ")
    includes = input("Enter rent includes (e.g., bills included): ")
    desc = input("Enter short property description: ")
    rent = float(input("Enter actual weekly rent (£): "))

    check_anomaly({
        'location': loc,
        'property_type': ptype,
        'rent_includes': includes,
        'description': desc,
        'weekly_rent': rent
    })

except Exception as e:
    print(f"Error during rental check: {e}")

