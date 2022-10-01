import requests
import os
import pandas as pd


def main():
    path_of_infiles = 'episode_links_by_season'
    character_data = []
    script_data = []

    for filename in os.listdir(path_of_infiles):
        if filename in ('season7.txt', 'season8.txt', 'season9.txt'):
            break
        print(f'Currently working on {filename}')
        with open(os.path.join(path_of_infiles, filename), 'r', encoding='utf-8') as f:
            links = f.readlines()
            for link in links:
                doc = requests.get(link).text
                doc = ''.join(doc.split('\n'))
                for line in doc.split('<p>'):
                    line = line[:-4].replace('<strong>', '').replace('</strong>', '')
                    try:
                        if line.split(':')[0][0].isalpha():
                            # split only on first occurrence. this will separate character name
                            data_lst = line.split(':', 1)
                            if len(data_lst) > 1:
                                character_data.append(data_lst[0])
                                script_data.append(data_lst[1].strip())
                    except IndexError:
                        continue

    outfile = pd.DataFrame({'text': script_data, 'character': character_data})
    outfile.to_csv('HIMYM_data_all_characters.csv')


if __name__ == "__main__":
    main()

