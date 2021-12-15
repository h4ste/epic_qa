import argparse
import json
import re
import pathlib
import logging
from typing import Optional, Sequence

import convert_xml
import spacy

from tld import get_tld
from tqdm.auto import tqdm

logger = logging.getLogger()
covid_regex = re.compile(
    r'\b(?:[Cc][Oo][Vv][Ii][Dd]|[Cc]orona|[Cc]oronavirus(?:es)?|nCoV|SARS|MERS|'
    r'[Ss]evere\w+[Aa]cute\w+[Rr]espiratory\w+[Ss]yndrome|'
    r'[Mm]iddle\w+[Ee]ast\w+[Rr]espiratory\w+[Ss]yndrome|'
    r'[Ww]uhan)\b')


def load_warc_file(warc_file):
    docs = []
    with open(warc_file) as in_file:
        pages = json.load(in_file)
    for page in tqdm(pages):
        sentences = page['sentences']
        if not sentences or len(sentences) <= 1:
            continue

        title = page['title']
        if not title or not covid_regex.search(page['title']):
            continue

        text = ' '.join(sentences)
        if not covid_regex.search(text):
            continue

        doc_id = page['WARC-Record-ID']
        if not doc_id.startswith('urn:uuid:'):
            continue
        doc_id = doc_id[9:]

        context_id = doc_id + '-C000'
        context = convert_xml.Context(text=text, context_id=context_id)

        end = 0
        for idx, sentence in enumerate(sentences):
            start = text.find(sentence, end)
            assert start >= 0, \
                f'In {warc_file}:{doc_id}, sentence |{sentence}| not found after |{idx}| in |{text}|'
            end = start + len(sentence)
            assert text[start:end] == sentence, \
                f'In {warc_file}:{doc_id}, sentence |{sentence}| did not match substring |{text}|'
            assert len(sentence) < 4096, \
                f'In {warc_file}:{doc_id}, Sentence {idx} was too long! ({len(sentence)} characters): |{sentence}'
            context.sentences.append(convert_xml.Sentence(start, end, sentence_id=f'{context_id}-{idx:0>3d}'))

        doc = convert_xml.Document(title=title,
                                   url=page['url'],
                                   document_id=doc_id,
                                   contexts=[context])
        docs.append(doc)
    return docs


def main(files: Sequence[pathlib.Path],
         output_dir: pathlib.Path,
         spacy_model: str = 'en_core_web_sm',
         valid_flds: Optional[pathlib.Path] = None):
    logger.debug('Loading spacy model %s...', spacy_model)
    nlp = spacy.load(spacy_model)

    if valid_flds:
        with open(valid_flds, 'r') as in_file:
            valid_flds = frozenset(in_file.read().splitlines())
            logging.info("Loaded %d valid first-level domains", len(valid_flds))

    def convert_file(warc_file: pathlib.Path):
        assert warc_file.is_file()

        for doc in load_warc_file(warc_file):
            if not doc.contexts:
                logging.warning('Skipping document %s with no contexts', doc)
                continue

            res = get_tld(doc.url, fix_protocol=True, fail_silently=True, as_object=True)
            if not res:
                continue
            tld = res.tld
            fld = res.fld
            if tld != "edu" and tld != "gov" and valid_flds and fld not in valid_flds:
                continue
            # Use spaCy to split sentences
            # convert_xml.split_sentences(nlp, doc, max_sent_len=40)

            # Write JSON file
            warc_dir = output_dir / fld
            warc_dir.mkdir(parents=True, exist_ok=True)
            output_file = warc_dir / f'{doc.document_id}-{warc_file.name[:-10]}.json'
            with open(output_file, 'w') as out_file:
                json.dump(doc.as_dict(), out_file, sort_keys=False, indent=4, ensure_ascii=True)

    def convert_directory(directory: pathlib.Path, root: Optional[pathlib.Path] = None):
        assert directory.is_dir()
        relative_name = directory.relative_to(root) if root else directory

        for file_ in tqdm(directory.iterdir(), desc=f'Converting {relative_name}', unit='files'):
            if file_.is_dir():
                convert_directory(file_)
            else:
                convert_file(file_)

    for file in tqdm(files, unit='file', desc='Converting data'):
        # If this file is a directory, we need to recurse into it
        if file.is_dir():
            convert_directory(file)
        elif file.is_file():
            convert_file(file)
        else:
            logger.error('Ignoring non-file %s', file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=pathlib.Path, help='warc-json file to process')
    parser.add_argument('--output_dir', required=True, type=pathlib.Path, help='folder to drop output file(s)')
    parser.add_argument('--spacy_model', nargs='?', type=str, default='en_core_web_sm',
                        help='spaCy model used for sentence splitting.')
    parser.add_argument('--valid_flds', nargs='?', type=pathlib.Path, default=None,
                        help='file with line-separated list of valid first-level domains')
    args = parser.parse_args()
    main(files=args.files,
         output_dir=args.output_dir,
         spacy_model=args.spacy_model,
         valid_flds=args.valid_flds)
