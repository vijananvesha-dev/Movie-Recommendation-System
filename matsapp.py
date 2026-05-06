import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from model import recommend_movie, movie_matrix

st.title("🎬 Movie Recommendation System")

# Dropdown
movie_list = movie_matrix.index.tolist()
selected_movie = st.selectbox("Select a Movie", movie_list)

# Recommend Button
if st.button("Recommend"):

    # Recommendations
  
    results = recommend_movie(selected_movie)

    st.write("###  Recommended Movies:")
    if results:
        for movie in results:
            st.write(movie)
    else:
        st.write("⚠️ No recommendations found.")

    # -------------------------------
    # 📊 Visualization 1: Top Movies
    # -------------------------------
    st.write("### 📊 Top Movies by Rating Count")

    rating_count = movie_matrix.sum(axis=1)

    if rating_count.empty:
        st.write("⚠️ No rating data available.")
    else:
        rating_count = rating_count.sort_values(ascending=False).head(10)

        plt.figure()
        rating_count.plot(kind='bar')
        plt.xlabel("Movies")
        plt.ylabel("Total Ratings")
        plt.title("Top 10 Most Rated Movies")

        st.pyplot(plt)

    # -------------------------------
    # 📈 Visualization 2: Similarity
    # -------------------------------
    st.write("### 📈 Similarity with Selected Movie")

    try:
        similarity = movie_matrix.corrwith(movie_matrix.loc[selected_movie])
        similarity = similarity.dropna()

        if similarity.empty:
            st.write("⚠️ No similarity data found for this movie.")
        else:
            similarity = similarity.sort_values(ascending=False).head(10)

            plt.figure()
            similarity.plot(kind='bar')
            plt.xlabel("Movies")
            plt.ylabel("Similarity Score")
            plt.title(f"Top Similar Movies to {selected_movie}")

            st.pyplot(plt)

    except Exception as e:
        st.write("⚠️ Error while calculating similarity.")
        st.write(e)