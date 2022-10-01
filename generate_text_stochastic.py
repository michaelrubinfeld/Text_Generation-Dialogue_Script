from project_tools import ProjectTools
from stochastic_text_generators import StochasticTextGenerator

tools = ProjectTools()
df = tools.clean_data('HIMYM_data_all_characters.csv')
robin_series = tools.get_data_of_characters(df, ['Robin'])  # trained on Robin's data

gen = StochasticTextGenerator(robin_series)
print(gen.naive_chain('Barney', length=10))
print(gen.markov_chain('Have you', 2))