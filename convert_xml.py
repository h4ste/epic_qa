import argparse
import hashlib
import json
import logging
import pathlib
import abc
import csv
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, List, Mapping, Union, AnyStr, Any, Callable
from xml.etree import ElementTree

import spacy
from tqdm.auto import tqdm


logger = logging.getLogger()


def get_hash(texts: Sequence[str]):
    # Sanity check because python doesn't distinguish between chars and strings,
    # and we want a sequence of real strings not a sequence of characters-as-strings
    assert not isinstance(texts, str)

    # Hash code is 40 character SHA-1
    hasher = hashlib.sha1()
    for text in texts:
        hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()[:40]


@dataclass
class Sentence:
    start: int
    end: int
    sentence_id: str = None

    def as_dict(self):
        return {
            'sentence_id': self.sentence_id,
            'start': self.start,
            'end': self.end
        }


@dataclass
class Context:
    text: str
    section: Optional[str] = None
    context_id: str = None
    sentences: List[Sentence] = field(default_factory=list)

    def as_dict(self):
        return {
            'section': self.section,
            'context_id': self.context_id,
            'text': self.text,
            'sentences': [sentence.as_dict() for sentence in self.sentences]
        }


@dataclass
class Document:
    title: str
    url: str
    document_id: str = None
    contexts: List[Context] = field(default_factory=list)

    def as_dict(self):
        return {
            'document_id': self.document_id,
            'metadata': {
                'title': self.title,
                'url': self.url
            },
            'contexts': [context.as_dict() for context in self.contexts],
        }


class DocumentIdentifier(abc.ABC):

    @abc.abstractmethod
    def get_id(self, doc: Document) -> str:
        pass


class ContentHashIdentifier(DocumentIdentifier):

    def get_id(self, doc: Document) -> str:
        return get_hash([context.text for context in doc.contexts])


class StaticUrlIdentifier(DocumentIdentifier):

    @classmethod
    def from_file(cls, url2ids_file: Union[str, bytes, os.PathLike]):
        url2ids = {}
        with open(url2ids_file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for url, doc_id in csv_reader:
                assert isinstance(doc_id, str)
                url2ids[url] = doc_id
        return cls(url2ids)

    def __init__(self, url2ids: Mapping[str, str]):
        self.url2ids = url2ids

    def get_id(self, doc: Document) -> str:
        return self.url2ids[doc.url]


class StaticFallbackUrlIdentifier(StaticUrlIdentifier):

    def __init__(self, url2ids: Mapping[str, str]):
        super().__init__(url2ids)
        self.new_url2ids = {}

    def get_id(self, doc: Document) -> str:
        if doc.url in self.url2ids:
            return self.url2ids[doc.url]
        else:
            doc_id = get_hash([context.text for context in doc.contexts])
            logger.debug("Generating ID %s for %s", doc_id, doc.url)
            self.new_url2ids[doc.url] = doc_id
            return doc_id

    def save_url2ids(self, url2ids_file, ids: Optional[str] = "new"):
        if ids == "new":
            url2ids = self.new_url2ids
        elif ids == "orig":
            url2ids = self.url2ids
        elif ids == "all":
            url2ids = {**self.url2ids, **self.new_url2ids}
        else:
            raise ValueError("Unsupported ID source. Expected one of \"new\", \"orig\" or \"all\".")

        with open(url2ids_file, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            for url, doc_id in url2ids.items():
                csv_writer.writerow([url, doc_id])


def get_text(element: ElementTree.Element) -> str:
    return ElementTree.tostring(element, encoding='utf-8', method='text').decode('utf-8').strip()


def read_xml(doc_file, section_joiner=' - ', doc_identifier: DocumentIdentifier = ContentHashIdentifier()) -> Document:
    # Parse xml
    tree = ElementTree.parse(doc_file)
    root = tree.getroot()
    assert root.tag == 'doc', 'Encountered unexpected root tag %s' % (root.tag)

    # Prepare initial document
    title = root.findtext('title')
    doc = Document(url=root.attrib['url'] or root.attrib['id'], title=title.strip() if title else None)

    # Some documents have text before any sections, we refer to these as 'root' context
    for root_text in (root.findall('text') + root.findall('summary')):
        text = get_text(root_text)
        if text:
            doc.contexts.append(Context(text=text))

    def process_section(section: ElementTree.Element, path: List[str], depth: int):
        assert section.tag == 'section', 'Encountered unexpected section tag %s' % (section.tag)
        if depth > 3:
            logger.warning('Processing section at depth %d in document %s', depth, doc_file)

        # Combine current section title with titles of its parent sections (if any)
        section_title = section.findtext('title')
        if section_title:
            section_title = section_title.strip()
            path.append(section_title)
        title = section_joiner.join(path)

        for section_text in (section.findall('text') + section.findall('summary')):
            text = get_text(section_text)
            if text:
                doc.contexts.append(Context(text=text, section=title))

        # Check child sections
        for sections_ in section.findall('sections'):
            for child in sections_.findall('section'):
                process_section(child, path, depth + 1)

    # Recursively process sections
    for sections in root.findall('sections'):
        for section_ in sections.findall('section'):
            process_section(section_, [], 0)

    # Add document and context IDs
    doc.document_id = doc_identifier.get_id(doc)
    for ctx_id, context in enumerate(doc.contexts):
        context.context_id = f'{doc.document_id}-C{ctx_id:0>3d}'

    return doc


def split_sentences(nlp,
                    doc: Document,
                    context_fn: Callable[[Document], List[Context]] = lambda doc: doc.contexts,
                    ncols: Union[None, int, str] = None,
                    max_sent_len: Optional[int] = None) -> None:
    for context in tqdm(context_fn(doc), desc=f'Sentencizing {doc.url}', leave=False, unit='context', ncols=ncols):
        ctx = nlp(context.text)
        ctx_id = context.context_id

        def make_sentence(sent_, sent_id):
            start, end = sent_.start_char, sent_.end_char

            # Sanity check that our character boundaries are reasonable
            if context.text[start:end] != sent_.text:
                logger.warning('Document %s, raw sentence:chars::[%d-%d)=|%s|',
                               doc.url, start, end, context.text[start:end])
                logger.error('Document %s, spacy sentence:tok::[%d-%d)=|%s|',
                             doc.url, sent_.start, sent_.end, sent_.text)
                raise AssertionError

            context.sentences.append(Sentence(start=start, end=end, sentence_id=f'{ctx_id}-S{sent_id:0>3d}'))

        sent_idx = 0
        for sent in ctx.sents:
            if max_sent_len and len(sent) > max_sent_len:
                orig_start, orig_end = sent.start_char, sent.end_char
                spans = []
                for start_ in range(sent.start, sent.end, max_sent_len):
                    span = ctx[start_:min(sent.end, start_ + max_sent_len)]
                    make_sentence(span, sent_idx)
                    spans.append(span)
                    sent_idx += 1
                if orig_start != spans[0].start_char or orig_end != spans[-1].end_char:
                    logger.error('Document %s, splitting sentence failed for |%s|, made %s',
                                 doc.url, context.text[orig_start:orig_end],
                                 [context.text[span.start_char:span.end_char] for span in spans])
                    raise AssertionError
            else:
                make_sentence(sent, sent_idx)
                sent_idx += 1
            #
            #
            # start, end = sent.start_char, sent.end_char
            #
            # # Sanity check that our character boundaries are reasonable
            # if context.text[start:end] != sent.text:
            #     logger.warning('Document %s, raw sentence:chars::[%d-%d)=|%s|',
            #                    doc.url, start, end, context.text[start:end])
            #     logger.error('Document %s, spacy sentence:tok::[%d-%d)=|%s|',
            #                  doc.url, sent.start, sent.end, sent.text)
            #     raise AssertionError
            #
            # # Warn if our sentence is crazy
            # if max_sent_len and (sent.end - sent.start) > max_sent_len:
            #     for start_ in range(sent.start, sent.end, max_sent_len):
            #         span = context.text[start_:start_ + max_sent_len]
            #         start, end = span.start_char, span.end_char
            #         context.sentences.append(Sentence(start=start, end=end, sentence_id=f'{ctx_id}-S{sent_id:0>3d}'))
            # else:
            #     context.sentences.append(Sentence(start=start, end=end, sentence_id=f'{ctx_id}-S{sent_id:0>3d}'))


def main(files: Sequence[pathlib.Path],
         output_dir: pathlib.Path,
         base_dir: Optional[pathlib.Path] = None,
         spacy_model: str = 'en_core_web_sm',
         section_joiner: str = ' - ',
         url2id_file: Optional[pathlib.Path] = None,
         fallback_url2id_file: Optional[pathlib.Path] = None):

    logger.debug('Loading spacy model %s...', spacy_model)
    nlp = spacy.load(spacy_model)

    if fallback_url2id_file:
        doc_identifier = StaticFallbackUrlIdentifier.from_file(url2id_file)
    elif url2id_file:
        doc_identifier = StaticUrlIdentifier.from_file(url2id_file)
    else:
        doc_identifier = ContentHashIdentifier()

    # noinspection PyShadowingNames
    def convert_file(file: pathlib.Path):
        assert file.is_file()
        doc = read_xml(file, section_joiner=section_joiner, doc_identifier=doc_identifier)

        if not doc.contexts:
            logging.warning('Skipping document %s with no contexts', doc)
            return

        # Use spaCy to split sentences
        split_sentences(nlp, doc)

        # Manage intermediate folders
        output_parent = (output_dir / file.parent.relative_to(base_dir)) if base_dir else output_dir
        if output_parent not in created_directories:
            # Save a bit of time by skipping directory creation for parents we've already made
            logging.debug('Creating directory %s', output_parent)
            output_parent.mkdir(parents=True, exist_ok=True)
            created_directories.add(output_parent)

        # Write JSON file
        output_file = output_parent / f'{doc.document_id}.json'
        with open(output_file, 'w') as out_file:
            json.dump(doc.as_dict(), out_file, sort_keys=False, indent=4, ensure_ascii=True)

    def convert_directory(directory: pathlib.Path):
        assert directory.is_dir()
        relative_name = directory.relative_to(base_dir) if base_dir else directory

        for file_ in tqdm(directory.iterdir(), desc=f'Converting {relative_name}', unit='files'):
            if file_.is_dir():
                convert_directory(file_)
            else:
                convert_file(file_)

    logger.debug('Starting file processing...')
    created_directories = set()
    for file in tqdm(files, unit='file', desc='Converting data'):
        # If this file is a directory, we need to recurse into it
        if file.is_dir():
            convert_directory(file)
        elif file.is_file():
            convert_file(file)
        else:
            logger.error('Ignoring non-file %s', file)

    logger.info("Writing new URL2ID mapping at %s", fallback_url2id_file)
    if fallback_url2id_file:
        assert isinstance(doc_identifier, StaticFallbackUrlIdentifier)
        doc_identifier.save_url2ids(fallback_url2id_file)


# noinspection PyTypeChecker
def parse_args() -> Mapping[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, type=pathlib.Path,
                        help='xml files to process')
    parser.add_argument('--base_dir', nargs='?', type=pathlib.Path, default=None,
                        help='if specified, input files with be relativized to --base_dir '
                             'and any intermediate folders will be preserved in --output_dir')
    parser.add_argument('--output_dir', required=True, type=pathlib.Path,
                        help='folder to drop output files')
    parser.add_argument('--spacy_model', nargs='?', type=str, default='en_core_web_sm',
                        help='spaCy model used for sentence splitting.')
    parser.add_argument('--section_joiner', nargs='?', type=str, default=' - ',
                        help='string to use when combining nested section titles')
    parser.add_argument('--url2id_file', nargs='?', type=pathlib.Path, default=None,
                        help='if specified, documents will be assigned IDs based on their url using the given '
                             'file, rather than ID\'s based on their contexts\' checksums')
    parser.add_argument('--fallback_url2id_file', nargs='?', type=pathlib.Path, default=None,
                        help='if specified, documents who were not in the original url2id file will have their '
                             'IDs generated from their content and the new IDs will be saved in the given file')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    main(**args)
