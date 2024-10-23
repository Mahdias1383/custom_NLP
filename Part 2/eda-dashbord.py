import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import altair as alt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

train = pd.read_csv(r'train_data.csv', delimiter=',')
title_brand = pd.read_csv(r'title_brand.csv', delimiter=',')

train['vote'] = train['vote'].str.replace(',', '')
train['vote'] = pd.to_numeric(train['vote'], errors='coerce')

train['review_length'] = train['reviewText'].str.len()


def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=15)
    plt.axis('off')
    st.pyplot(plt)


# st.markdown("# EDA (Exploratory Data Analysis)")

st.sidebar.markdown("## EDA (Exploratory Data Analysis)")
page = st.sidebar.radio("", ["• Page 1: Data Analysis", "• Page 2: Word Clouds", "• Page 3: Top Reviewers",
                             "• Page 4: Review Length Distribution", "• Page 5: Best Products", "• Page 6: Top Brands"])

if page == "• Page 1: Data Analysis":
    st.markdown("""
    ### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> 1.توزیع ستون</div>
    """, unsafe_allow_html=True)

    st.write("### Train Data Summary")
    st.write(train.describe())

    rating_filter = st.slider("Filter Ratings", min_value=int(train['overall'].min()),
                              max_value=int(train['overall'].max()),
                              value=(int(train['overall'].min()), int(train['overall'].max())))

    filtered_data = train[train['overall'].between(rating_filter[0], rating_filter[1])]

    chart = alt.Chart(filtered_data).mark_bar(color='skyblue').encode(
        x=alt.X('overall:O', title='Rating', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count():Q', title='Count'),
        tooltip=['overall', 'count()']
    ).properties(
        width=600,
        height=400,
        title='Distribution of Overall Ratings'
    ).interactive()

    st.altair_chart(chart)
    st.markdown(
        """### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> از آنجا که توزیع متوازن نیست میتوانیم با استفاده از روش های مختلفی داده را آماده آموزش کنیم</div> """
        , unsafe_allow_html=True)

elif page == "• Page 2: Word Clouds":
    st.markdown("""
    ### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> 2. رسم ابر کلمات</div>
    """, unsafe_allow_html=True)

    positive_reviews = ' '.join(train[train['overall'].isin([4, 5])]['reviewText'].dropna())
    neutral_reviews = ' '.join(train[train['overall'] == 3]['reviewText'].dropna())
    negative_reviews = ' '.join(train[train['overall'].isin([1, 2])]['reviewText'].dropna())

    create_word_cloud(positive_reviews, 'Positive Reviews')
    create_word_cloud(neutral_reviews, 'Neutral Reviews')
    create_word_cloud(negative_reviews, 'Negative Reviews')

elif page == "• Page 3: Top Reviewers":
    st.markdown("""
    ### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> 3.نمایش ۱۰ نفری که در مجموع نظرات‌شان بیشتر مفید واقع شده</div>
    """, unsafe_allow_html=True)

    best_reviewers = train.groupby('reviewerID')['vote'].sum().reset_index()

    top10_reviewers = best_reviewers.sort_values(by='vote', ascending=False).head(10)

    st.write("### Top 10 Reviewers by Votes")
    st.dataframe(top10_reviewers[['reviewerID', 'vote']])

elif page == "• Page 4: Review Length Distribution":
    st.markdown("""
    ### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> 4.هیستوگرام طول نظرات</div>
    """, unsafe_allow_html=True)

    original_hist = alt.Chart(train).mark_bar(color='skyblue').encode(
        alt.X('review_length:Q', bin=alt.Bin(maxbins=50), title='Review Length (characters)'),
        alt.Y('count()', title='Frequency'),
        tooltip=['count()']
    ).properties(
        width=600,
        height=400,
        title='Histogram of Review Length (Original)'
    ).interactive()

    st.altair_chart(original_hist)

    filtered_train = train[(train['review_length'] > 15) & (train['review_length'] < 5000)]

    filtered_hist = alt.Chart(filtered_train).mark_bar(color='green').encode(
        alt.X('review_length:Q', bin=alt.Bin(maxbins=50), title='Review Length (characters)'),
        alt.Y('count()', title='Frequency'),
        tooltip=['count()']
    ).properties(
        width=600,
        height=400,
        title='Histogram of Review Length (Filtered)'
    ).interactive()

    st.altair_chart(filtered_hist)

elif page == "• Page 5: Best Products":
    st.markdown("""
    ### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> 5.محصولاتی که بیشترین امتیاز ۵ را کسب کرده‌انده</div>
    """, unsafe_allow_html=True)

    merged_df = train.merge(title_brand[['asin', 'title', 'brand']], on='asin', how='inner')

    five_stars = merged_df[merged_df['overall'] == 5]

    best_products = five_stars.groupby('asin').size().reset_index(name='5star_count')

    best_products = best_products.sort_values(by='5star_count', ascending=False).head(10)

    best_products = best_products.merge(merged_df[['asin', 'title', 'brand']].drop_duplicates(), on='asin', how='left')

    st.write("### Top 10 Products by 5-Star Ratings")
    st.dataframe(best_products[['title', 'brand', '5star_count']])

elif page == "• Page 6: Top Brands":
    st.markdown("""
        ### <div dir="rtl" style="font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 20px; color: #ffffff; font-weight: bold; padding: 10px; border-radius: 8px; background: linear-gradient(135deg, #2EA149, #60a5fa); text-align: right;"> 6.میانگین نظرات 10 برندی که بیشترین تعداد نظرات را داشتند</div>
        """, unsafe_allow_html=True)

    merged_df = train.merge(title_brand[['asin', 'title', 'brand']], on='asin', how='inner')
    brand_review_counts = merged_df.groupby('brand').size().reset_index(name='review_count')

    top10_brands = brand_review_counts.sort_values(by='review_count', ascending=False).head(10)

    brand_avg_rate = merged_df[merged_df['brand'].isin(top10_brands['brand'])].groupby('brand')[
        'overall'].mean().reset_index(name='average_rate')

    top_10_avg = top10_brands.merge(brand_avg_rate, on='brand')

    top_10_avg = top_10_avg.sort_values(by='average_rate', ascending=False)

    st.write("### Top 10 Brands by Average Rating")
    st.dataframe(top_10_avg[['brand', 'average_rate']])
