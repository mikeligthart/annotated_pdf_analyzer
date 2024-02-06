import os
from itertools import zip_longest
from os.path import join
from shutil import copyfile
from typing import List, Tuple
import fitz


class ExtractAnnotationsFromPDF:

    def __init__(self, path_to_source_folder):
        self.path_to_source_folder = path_to_source_folder

    def extract_annotations(self, exclude: list):
        annotations = []
        for pdf in os.listdir(self.path_to_source_folder):
            participant_id = int(pdf.split('.')[0])
            if participant_id in exclude:
                print(f'Participant {participant_id} is on exclusion list.')
            else:
                print(f'Processing participant: {participant_id}')
                annotations.append({
                    'participant_id': participant_id,
                    'annotations': self._extract_annotations_in_pdf(join(self.path_to_source_folder, pdf))
                })
        return annotations

    def _extract_annotations_in_pdf(self, path_to_test_pdf):
        doc = fitz.open(path_to_test_pdf)

        highlights = []
        comments = []
        for page in doc:
            highlights += self._extract_highlights_from_page(page)  # extract highlighted text
            for annot in page.annots():
                comments.append(annot.info["content"])  # extract text from comments

        return list(zip_longest(highlights, comments))

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
        wordlist.sort(key=lambda w: (w[3], w[0]))  # ascending y, then x

        highlights = []
        annot = page.first_annot
        while annot:
            if annot.type[0] == 8:
                highlights.append(self._parse_highlight(annot, wordlist))
            annot = annot.next
        return highlights

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


if __name__ == '__main__':
    analyser = ExtractAnnotationsFromPDF('input/')
    annotations = analyser.extract_annotations(exclude=[52])
    # annotations = analyser._extract_annotations_in_pdf('input/1.pdf')
    print(annotations)

