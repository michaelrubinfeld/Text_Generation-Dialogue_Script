import pandas as pd
import re
import numpy as np

class ProjectTools:

    def __init__(self):
        self.df = pd.DataFrame()

    @staticmethod
    def _remove_director_notes(text):
        return re.sub(r"[\(\[].*?[\)\]]", "", text)   # remove everything inside parentheses

    def clean_data(self, url):
        self.df = pd.read_csv(url)    # get data
        self.df = self.df.iloc[:, 1:]   # remove redundant column
        self.df = self.df.astype(str)
        self.df = self.df[self.df['character'].str.match(r'\A[\w-]+\Z')]  # keep only valid character names

        # remove leftover html stuff
        self.df['character'] = self.df['character'].apply(lambda x: x.lower().capitalize())
        self.df['text'] = self.df['text'].apply(lambda x: x.split('<')[0])
        self.df['text'] = self.df['text'].apply(lambda x: x.replace('&quot;', '').replace('"', ''))
        self.df['text'] = self.df.text.map(lambda x: self._remove_director_notes(x))

        return self.df

    @staticmethod
    def get_data_of_characters(df, characters):
        for character in characters:
            if character not in ['Marshall', 'Ted', 'Barney', 'Lily', 'Robin']:
                raise ValueError('Character must be a main character in HIMYM')

        try:
            return df.loc[df.character.isin(characters)].reset_index(drop=True)['text']
        except KeyError as e:
            print(f'Column {e} missing when calling get_data_of_character')
