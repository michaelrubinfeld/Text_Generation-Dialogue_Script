import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt

df = pd.read_csv('HIMYM_data_all_characters.csv')

# Who talks the most?
char_counts = df['character'].value_counts().reset_index().rename(columns={'index':'char','character':'count'}).head(20)
fig = px.bar(char_counts, x="count", y="char", color="char", title="Characters Lines Count")
fig.show()


# Wordcloud
def wordcload_by_char(df,character_name=None):
    df['text'] = df['text'].astype(str)
    if character_name:
        df = df[df['character']==character_name]
    else:
        character_name = 'All characters'
    text = ' '.join(df['text'])
    wordcloud = WordCloud(width=500, height=400, prefer_horizontal=1, max_font_size=100, max_words=50,
                          background_color="white").generate(str(text))
    plt.imshow(wordcloud)
    plt.axis("off")
    wordcloud_plt = plt.title(character_name + " WordCloud")
    plt.show()

wordcload_by_char(df,character_name="Barney")
wordcload_by_char(df,character_name="Robin")
wordcload_by_char(df)