import os
from itertools import zip_longest
from os.path import join
from shutil import copyfile
from typing import List, Tuple
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy
nltk.download('stopwords')
nltk.download('punkt')
import spacy
import re
import spacy.cli
#spacy.cli.download("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()




class ExtractAnnotationsFromPDF:

    def __init__(self, path_to_source_folder):
        self.path_to_source_folder = path_to_source_folder

    def extract_annotations(self, exclude: list):
        annotations = []
        for pdf in os.listdir(self.path_to_source_folder):
            participant_id = int(pdf.split('.')[0])
            doc = fitz.open(self.path_to_source_folder + '/' + pdf)
            page = doc.load_page(3 - 1)
            title = page.get_text().split('\n', 1)[0]
            if participant_id in exclude:
                print(f'Participant {participant_id} is on exclusion list.')
            else:
           #     print(f'Processing participant: {participant_id}')
                annotations.append({
                    'participant_id': participant_id,
                    'title': title,
                    'annotations': self._extract_annotations_in_pdf(join(self.path_to_source_folder, pdf))
                })
        return annotations

    def _extract_annotations_in_pdf(self, path_to_test_pdf):
        doc = fitz.open(path_to_test_pdf)

        highlights = []
        comments = []
       # for page in doc:
        # for page_num in range(2, doc.page_count):
        #     page = doc.load_page(page_num)

        #     highlights += self._extract_highlights_from_page(page)  # extract highlighted text
        #     for annot in page.annots():
        #         comments.append(annot.info["content"])  # extract text from comments

        for page_num in range(2, doc.page_count):
            page = doc.load_page(page_num)

            highlights += self._extract_highlights_from_page(page)  # extract highlighted text
            annotations = [(annot, annot.rect.y0) for annot in page.annots()]  # Get annotations with their vertical positions
            annotations.sort(key=lambda x: x[1])  # Sort annotations based on their vertical positions

            for annot, _ in annotations:
                comments.append(annot.info["content"])  # extract text from comments

        data = list(zip_longest(highlights, comments)) #old return

        cleaned_data = []
        i = 0
        while i < len(data):
            sentence, highlight = data[i]
            if sentence != '' and highlight == '':
                if i + 1 < len(data):
                    next_sentence, next_highlight = data[i + 1]
                    if next_sentence == '' and next_highlight != '':
                        cleaned_data.append((sentence, next_highlight))
                        i += 1  # skip the next sentence and highlight as they have been combined
            elif highlight != '' and sentence == '':
                if i + 1 < len(data):
                    next_sentence, next_highlight = data[i + 1]
                    if next_sentence != '' and next_highlight == '':
                        cleaned_data.append((next_sentence, highlight))
                        i += 1  # skip the next sentence and highlight as they have been combined
            elif sentence != '' and highlight != '':
                cleaned_data.append((sentence, highlight))
            i += 1

        return cleaned_data
     #   return list(zip_longest(highlights, comments))

    def _parse_highlight(self, annot: fitz.Annot,
                         wordlist: List[Tuple[float, float, float, float, str, int, int, int]]) -> str:
        """ Parse a single highlight"""
        points = annot.vertices
        quad_count = int(len(points) / 4)
        sentences = []
        for i in range(quad_count):
            # where the highlighted part is
            r = fitz.Quad(points[i * 4: i * 4 + 4]).rect

            words = [w for w in wordlist if fitz.Rect(w[:4]).intersects(r)]
            sentences.append(" ".join(w[4] for w in words))
        sentence = " ".join(sentences)
        return sentence

    def _extract_highlights_from_page(self, page):
        """ Extract all highlights from a page"""
        wordlist = page.get_text("words")  # list of words on page
        wordlist.sort(key=lambda w: (w[1], w[0]))  # ascending y, then x
       # print(wordlist)

        highlights = []
        annot = page.first_annot #order is based on what highlights were added FIRST.
        while annot:
            if annot.type[0] == 8:
                highlights.append(self._parse_highlight(annot, wordlist))
            annot = annot.next

        # Extract all annotations from the page
        annotations = page.annots()

        # Sort annotations by the y-coordinate of the top-left corner of their bounding boxes
        sorted_annotations = sorted(annotations, key=lambda annot: annot.rect[3])

      #  print([self._parse_highlight(annot, wordlist) for annot in sorted_annotations]) #PROBABLY FIX!

        scenario_string = ' '.join(word_tuple[4] for word_tuple in wordlist)
        
        return [self._parse_highlight(annot, wordlist) for annot in sorted_annotations]

    # def _extract_highlights_from_page(self, page):
    #     """ Extract all highlights from a page"""
    #     highlights = []

    #     # Extract highlights from annotations
    #     annot = page.first_annot
    #     while annot:
    #         if annot.type[0] == 8:
    #             highlights.append(self._parse_highlight(annot))
    #         annot = annot.next

    #     # Sort highlights based on their vertical position (y-coordinate)
    #     highlights.sort(key=lambda h: h['bounding_box'][1])  # Sort by the y-coordinate of the top-left corner

    #     return highlights


    @staticmethod
    def select_files_with_consent(input_path, output_path):
        """Filters files with the opt-out box not selected and assigns them a participant number"""
        participant_number = 1
        for filename in os.listdir(input_path):
            print(filename)
            doc = fitz.open(join(input_path, filename))
            widgets = list(doc[0].widgets())
            if len(widgets) > 0 and not widgets[0].field_value:
                copyfile(join(input_path, filename), join(output_path, f'{participant_number}.pdf'))
                participant_number += 1



#TODO add all functions to class
# class ReadAndMarkPDF:
#     def __init__(self, path_to_source_folder):
#         self.path_to_source_folder = path_to_source_folder
    

def extract_text_from_pages(pdf_path, start_page):  # Extract text from a specific PDF page on.
    text = ""
    doc = fitz.open(pdf_path)
    
    for page_number in range(start_page, len(doc) + 1):
        page = doc.load_page(page_number - 1)
        text += page.get_text()
    
    doc.close()

    return text


def process_texts(folder_path, start_page): # Process complete texts
    full_texts = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pages(pdf_path, start_page)
            # Extract the first line as title
            first_line = text.split('\n', 1)[0].strip() 
            # Remove title from text
            text = text.replace(first_line,"",1)
            
            # Save unique version of the text
            if first_line not in full_texts:
                full_texts[first_line] = text
    
    return full_texts


def pair_highlight_to_text(annotations): # Separate the highlights based on the texts they are from TODO: Add participant_ids
    for annotation in annotations:
         # For now: remove annotations not linked to a sentence.. TODO fix this.
        annotation['annotations'] = [ann for ann in annotation['annotations'] if ann[0] is not None]
    separated_annotations = {}

    for item in annotations:
        # Remove any leading or trailing whitespace
        title = item['title'].strip()
        if title not in separated_annotations:
            # Initialize separate lists for i[0] and i[1]
            separated_annotations[title] = {'sentences': [], 'comments': []} #participants_ids
        
        # Append i[0] and i[1] to their respective lists
        separated_annotations[title]['sentences'].extend([i[0] for i in item['annotations']])
        separated_annotations[title]['comments'].extend([i[1] for i in item['annotations']])
    #    separated_annotations[title]['participants_ids'].extend([item['participant_id'] for _ in item['annotations']])



    for title, sentences in separated_annotations.items():
        # Split each sentence into separate items based on '.'
            split_sentences = []
            split_comments = []
       #     split_participant_ids =

            for i, sentence in enumerate(sentences['sentences']):
                # Split text into sentences
                split_items = [s.strip().replace("\\", "") for s in sentence.split('.')]
                 # Remove very short annotations.
                split_items = [item for item in split_items if item and not (len(item)<5)]
                split_sentences.extend(split_items)
                # Add comments for each sentence
                split_comments.extend([sentences['comments'][i]] * len(split_items))
     #           split_participant_ids.extend([sentences['comments'][i]] * len(split_items))


            separated_annotations[title]['sentences'] = split_sentences
            separated_annotations[title]['comments'] = split_comments

    return separated_annotations


def show_word_clouds(separated_annotations, stop_words): # Present word clouds for comments on each text (optional)
    for _, content in separated_annotations.items():
        # Tokenize comments into words and remove punctuation
        word_tokens = []
        for comment in content['comments']:
            # Tokenize
            tokens = word_tokenize(comment)
            # Remove punctuation
            tokens = [word.lower() for word in tokens if word.isalnum()]
            word_tokens.extend(tokens)

        # Remove stopwords
        filtered_words = [word for word in word_tokens if word not in stop_words]

        # Count the frequency of each word
        word_freq = {}
        for word in filtered_words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        # Display the word cloud
        # plt.figure(figsize=(10, 5))
        # plt.imshow(wordcloud, interpolation='bilinear')
        # plt.axis('off')
        # plt.show()


def html_input(title,text, separated_annotations): # Generate input for HTML, including the list of sentences, highlight count, and comments
        #  print(text) 
        #  sentence_list = [s.strip() + '.' for s in ''.join(text.splitlines()).split('.') if s.strip()]
         # Process the text with SpaCy
         doc = nlp(text)
        # Extract sentences
         sentence_list = [sent.text.strip() for sent in doc.sents]

         highlight_counts = np.zeros(len(sentence_list))
        # Create empty sublists using a list comprehension
         comments_indices = [[] for _ in range(len(sentence_list))]

         for annotation_index in range(len(separated_annotations[title]['sentences'])):
            for i in range(len(sentence_list)):
             if separated_annotations[title]['sentences'][annotation_index] in sentence_list[i]:
                 if separated_annotations[title]['comments'][annotation_index] not in [separated_annotations[title]['comments'][index] for index in comments_indices[i]]:
                    highlight_counts[i] += 1
                    comments_indices[i].append(annotation_index)

         comments_output = []
         for i, sentence in enumerate(sentence_list):
            comments = []
            if highlight_counts[i] > 0:
                for comment in comments_indices[i]:
                    if not separated_annotations[title]['comments'][comment].strip() is '':
                        comments.append(separated_annotations[title]['comments'][comment].strip())
                comments_output.append([sentence, str(int(highlight_counts[i])), comments])

         return sentence_list, highlight_counts, comments_output


# def generate_html(title, sentences, annotations, comments):     # Generate HTML code for the title
#     html_title = f'<h1>{title}</h1>'
    
#     # Generate HTML code for the sentences with highlighted background and black text
#     html_sentences = []
#     for sentence, num_annotations in zip(sentences, annotations):
#         # Calculate the brightness of the color based on the number of annotations
#         brightness = int((num_annotations / max(annotations)) * 255)
#         # Convert brightness to hexadecimal color code for highlight
#         highlight_color = "#{:02x}{:02x}{:02x}".format(255, 255-brightness, 255-brightness)  # White to red gradient
        
#         # Generate HTML code for the sentence with highlighted background and black text
#         html_sentence = f'<div><span style="background-color:{highlight_color};">{sentence}</span></div>'
#         html_sentences.append(html_sentence)
    
#     # Concatenate HTML-coded sentences into a single HTML string
#     html_output = html_title + '<div class="heatmap">' + ''.join(html_sentences) + '</div>'
    
#     # Add whitespace between sections
#     html_output += '<br><br>'
    
#     # Add subtitle for comments
#     html_output += '<h2>Comments on the highlighted areas</h2>'
    
#     # Generate HTML code for the comments for each sentence
#     for i, sentence_comments in enumerate(comments):
#         sentence_comment = sentence_comments[0]
#         num_times_marked = sentence_comments[1]
#         additional_comments = sentence_comments[2]

#         # Add section for each sentence
#         html_output += f'<div><h3>{i+1}. Sentence: {sentence_comment}</h3>'
#         html_output += f'<p>Times marked: {num_times_marked}</p>'
#         html_output += '<p>Comments:</p><ul>'
#         for comment in additional_comments:
#             html_output += f'<li>{comment}</li>'
#         html_output += '</ul></div><br>'
    
#     return html_output

def generate_html(title, sentences, annotations, comments):
    # Generate HTML code for the title
    html_title = f'<h1>{title}</h1>'
    
    # Generate HTML code for the sentences with highlighted background and black text
    html_sentences = []
    for sentence, num_annotations in zip(sentences, annotations):
        # Calculate the brightness of the color based on the number of annotations
        brightness = int((num_annotations / max(annotations)) * 255)
        # Convert brightness to hexadecimal color code for highlight
        highlight_color = "#{:02x}{:02x}{:02x}".format(255, 255-brightness, 255-brightness)  # White to red gradient
        
        # Generate HTML code for the sentence with highlighted background and black text
        html_sentence = f'<span style="background-color:{highlight_color};">{sentence}</span>'
        html_sentences.append(html_sentence)
    
    # Concatenate HTML-coded sentences into a single HTML string
    html_output = html_title + '<div class="heatmap">' + ' '.join(html_sentences) + '</div>'
    
    # Add whitespace between sections
    html_output += '<br><br>'
    
    # Add subtitle for comments
    html_output += '<h2>Comments on the highlighted areas</h2>'
    
    # Generate HTML code for the comments for each sentence
    for i, sentence_comments in enumerate(comments):
        sentence_comment = sentence_comments[0]
        num_times_marked = sentence_comments[1]
        additional_comments = sentence_comments[2]

        # Add section for each sentence
        html_output += f'<div><h3>{i+1}. Sentence: {sentence_comment}</h3>'
        html_output += f'<p>Times marked: {num_times_marked}</p>'
        html_output += '<p>Comments:</p><ul>'
        for comment in additional_comments:
            html_output += f'<li>{comment}</li>'
        html_output += '</ul></div><br>'
    
    return html_output

def create_dataframe(annotations):
    # Flatten the annotations
    flattened_annotations = []
    for d in annotations:
        for annotation in d['annotations']:
            flattened_annotations.append({'participant_id': d['participant_id'],
                                        'title_scenario': d['title'],
                                        'annotation_sentence': annotation[0],
                                        'annotation_comment': annotation[1]})

    # Create DataFrame
    df = pd.DataFrame(flattened_annotations)
    return df



if __name__ == '__main__':
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    analyser = ExtractAnnotationsFromPDF('input')
    annotations = analyser.extract_annotations(exclude=[52])
    print(annotations)
    info_dataframe =  create_dataframe(annotations)
    info_dataframe.to_csv('output/scenario_dataframe.csv', index=False)
    separated_annotations = pair_highlight_to_text(annotations)
    full_texts = process_texts('input/', 3)
    stop_words = set(stopwords.words('english'))          # Define stopwords to be removed
  #  show_word_clouds(separated_annotations, stop_words)



    for title, text in full_texts.items():
        sentence_list, highlight_counts, comments_output = html_input(title, text, separated_annotations)

        html_output = generate_html(title, sentence_list, highlight_counts, comments_output).encode('ascii', 'ignore').decode()

        html_filename = os.path.join(output_dir, f"{title}.html")
        png_filename = os.path.join(output_dir, f"{title}.png")

        with open(html_filename, "w") as f:
            f.write(html_output)
