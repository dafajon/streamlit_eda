from PIL.Image import radial_gradient
from numpy.lib.twodim_base import _trilu_dispatcher
import streamlit as st
import numpy as np
import pandas as pd 
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache()
def read_data(datapath: str, **kwargs):
    df = pd.read_csv(datapath, sep=kwargs.get("sep"))
    return df

def app():
    st.title("Mercari Price Suggestion Challenge 💰🔮 EDA")
    st.header("Case Description")
    
    st.markdown("Product pricing gets hard at scale, considering just how many products are sold online. Clothing has strong seasonal pricing trends and is heavily influenced by brand names, while electronics have fluctuating prices based on product specs. ")
    st.markdown("**Mercari**, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers.")
    st.image("images/logo.png")
    st.header("Import Dataset")


    df = read_data("data/train.tsv", sep="\t").sample(10_000, random_state=42)
    st.dataframe(df.head())

    st.text(f"No of intances: {df.shape[0]}")

    st.markdown("Display basic statistics of the numeric variables")
    st.write(df.describe())

    st.header("Detailed Dataset Stats")
    
    #exclude_cols=["train_id", "name", "item_description"]
    #df_for_pr = df[df.columns.difference(exclude_cols)]
    #pr = df_for_pr.profile_report()
    #st_profile_report(pr)

    st.header("Deeper Dive into Variables")

    st.subheader("Target Variable")

    st.code("df[\"price\"].min()")
    st.code(df["price"].min())
    st.code("df[\"price\"].max()")
    st.code(df["price"].max())

    sns.distplot(df.price)
    st.pyplot()

    st.markdown("The item price ranges from 0 (I guess some items on Mercari are given away?) to $2009. Let’s look at the histogram of prices. Because price is likely skewed and because there are some 0s, we’ll plot the log of price + 1.")

    sns.distplot(np.log1p(df.price))
    st.pyplot()

    st.subheader("Item Condition")

    sns.countplot(data=df, x="item_condition_id")
    st.pyplot()

    st.markdown("The item condition ranges from 1 to 5. There are more items of condition 1 than any other. Items of condition 4 and 5 are relatively rare.")

    df["log_price"] = df.price.transform(np.log1p)
    sns.boxplot(data=df, x="item_condition_id", y="log_price")
    st.pyplot()
    st.markdown("Condition 5 clearly has the highest price, however condition 1 has the next-highest price, followed by condition 2, then 3, then 4. Condition 1 is the best and 5 is the worst. Condition 5 is a bit of an anomaly in that it has the highest price. However, it also has the fewest number of items, so our point estimate has the most uncertainty.")
    
    st.subheader("Shipping")
    st.markdown("The `shipping` variable is a binary variable indicating whether the shipping for the item is paid for by the seller (1) or not (0).")

    sns.countplot(data=df, x="shipping")
    st.pyplot()

    sns.kdeplot(data=df, x="log_price", 
                hue="shipping",fill=True, 
                common_norm=False, 
                palette="crest",
                alpha=.5, linewidth=0)
    st.pyplot()

    st.markdown("Items where the shipping is paid by the seller have a lower average price.")

    st.subheader("Brand")

    st.code("""df.groupby(["brand_name"]).price.median().reset_index().rename(columns={"price":"median_price"}).sort_values(by="median_price", ascending=False).head(25)
            """)


    max_price_brand = df.groupby(["brand_name"]).price.median().reset_index().rename(columns={"price":"median_price"}).sort_values(by="median_price", ascending=False).head(25)
    sns.scatterplot(data=max_price_brand, y="brand_name", x="median_price")
    st.pyplot()

    st.subheader("Item Category")
    st.code("len(df[\"category_name\"].unique())")
    st.code(len(df["category_name"].unique()))

    category_price = df.groupby("category_name").price.median().reset_index().rename(columns={"price":"median_price"}).sort_values("median_price", ascending=False).reset_index(drop=True).head(25)
    category_count = df.groupby("category_name").train_id.count().reset_index().rename(columns={"train_id":"count"}).sort_values("count", ascending=False).reset_index(drop=True).head(25)

    sns.scatterplot(data=category_count, y="category_name", x="count")
    st.pyplot()
    st.markdown("#### Price by category")
    st.markdown("Looking at the ten most popular categories shows that women’s apparel is quite popular on Mercari. Of then top ten categories, 5 are women’s apparel. Makeup is also a highly listed category as are electronics. Now let’s examine prices by category. What are the product categories with the highest selling price?")
    sns.scatterplot(data=category_price, y="category_name", x="median_price")
    st.pyplot()

    st.subheader("Create category hieararchy")
    st.code("df['level_1_cat'] = df['category_name'].apply(lambda category: category.split('/')[0])")
    st.code("df['level_2_cat'] = df['category_name'].apply(lambda category: category.split('/')[1])")
    
    df = df[df.category_name.apply(lambda x: '/' in str(x))]
    df['level_1_cat'] = df['category_name'].apply(lambda category: str(category).split('/')[0])
    df['level_2_cat'] = df['category_name'].apply(lambda category: str(category).split('/')[1])

    lvl1_category_price = df.groupby("level_1_cat").price.median().reset_index().rename(columns={"price":"median_price"}).sort_values("median_price", ascending=False).reset_index(drop=True).head(25)
    lvl2_category_price = df.groupby("level_1_cat").price.median().reset_index().rename(columns={"price":"median_price"}).sort_values("median_price", ascending=False).reset_index(drop=True).head(25)

    st.subheader("Price by Level 1 Category")
    sns.boxplot(data=df, x="log_price", y="level_1_cat")
    st.pyplot()

    st.subheader("Price by Level 2 Category")
    fig, ax = plt.subplots(figsize=(10,30))
    sns.boxplot(data=df, x="log_price", y="level_2_cat", ax=ax)
    st.pyplot()

    st.subheader("Feature Interactions")

    st.markdown("#### Top Category by Condition")

    chart = alt.Chart(df).mark_circle().encode(
    x='item_condition_id:O',
    y='level_1_cat:O',
    size='count():Q'
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("Women’s items of condition 1,2, and 3 are the most numerous. This is followed by Beauty products.")

    st.markdown("#### Top Category by Condition - Item Prices")
    chart = alt.Chart(df).mark_circle().encode(
    x='item_condition_id:O',
    y='level_1_cat:O',
    size='median(price):Q'
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Brand Name by Condition")
    brand_df = df[df.brand_name.isin(list(max_price_brand.brand_name.unique()))]
    chart = alt.Chart(brand_df).mark_circle().encode(
    x='item_condition_id:O',
    y='brand_name:O',
    size='count():Q'
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("Women’s items of condition 1,2, and 3 are the most numerous. This is followed by Beauty products.")

    st.markdown("#### Brand Name by Condition - Item Prices")
    chart = alt.Chart(brand_df).mark_circle().encode(
    x='item_condition_id:O',
    y='brand_name:O',
    size='median(price):Q'
    )
    st.altair_chart(chart, use_container_width=True)


    st.subheader("Text Analysis")

    st.markdown("At this point, we’ve already done a significant amount of data exploration and we haven’t even gotten to to real bulk of this problem, the descritption text. The description in unstructured data, so in order to explore it fully, we’ll need to do some text processing and normalization.")

    stopwords = ["and", "is", "there", "then", 
                 "of", "so", "the", "in", "to", 
                 "with", "are", "will", "for", 
                 "it", "on"]

    st.markdown("#### Most Common Uni-grams")
    cv = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords)
    count_array = cv.fit_transform(df.item_description).toarray().sum(axis=0)
    word_array = cv.get_feature_names()
    vocab_dict = {"token": word_array, "count": count_array}
    vocab_df = pd.DataFrame(vocab_dict).sort_values("count", ascending=False)
    wc_dict = {}
    for row in vocab_df.iterrows():
        wc_dict[row[1]["token"]] = row[1]["count"]
    sns.barplot(data=vocab_df.head(15), x="count", y="token")
    wc = WordCloud().fit_words(wc_dict)
    
    st.image(wc.to_array(), use_column_width="always")
    st.pyplot()
    
    

    st.markdown("#### Most Common Bi-grams")
    cv = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords)
    count_array = cv.fit_transform(df.item_description).toarray().sum(axis=0)
    word_array = cv.get_feature_names()
    vocab_dict = {"token": word_array, "count": count_array}
    vocab_df = pd.DataFrame(vocab_dict).sort_values("count", ascending=False)
    wc_dict = {}
    for row in vocab_df.iterrows():
        wc_dict[row[1]["token"]] = row[1]["count"]
    sns.barplot(data=vocab_df.head(15), x="count", y="token")
    wc = WordCloud().fit_words(wc_dict)
    
    st.image(wc.to_array(), use_column_width="always")
    st.pyplot()

    st.markdown("#### Most Common Tri-grams")
    cv = CountVectorizer(ngram_range=(3, 3), stop_words=stopwords)
    count_array = cv.fit_transform(df.item_description).toarray().sum(axis=0)
    word_array = cv.get_feature_names()
    vocab_dict = {"token": word_array, "count": count_array}
    vocab_df = pd.DataFrame(vocab_dict).sort_values("count", ascending=False)
    wc_dict = {}
    for row in vocab_df.iterrows():
        wc_dict[row[1]["token"]] = row[1]["count"]
    sns.barplot(data=vocab_df.head(15), x="count", y="token")
    wc = WordCloud().fit_words(wc_dict)
    
    st.image(wc.to_array(), use_column_width="always")
    st.pyplot()



if __name__ == "__main__":
    app()