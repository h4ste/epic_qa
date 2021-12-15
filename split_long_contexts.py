"""
Creates output format for TAC-QA task
Splits long contexts with more than 15 sentences into 15 sentence chunks

"""

import os
import sys
import json
import tqdm
from typing import Dict, List
import multiprocessing
from argparse import ArgumentParser


def _chunkify(ctx: Dict, doc_id: str, cid_start: int, max_len: int) -> List[Dict]:
    """
    Break context into chunks of size max_len
    :param ctx: context dict
    :param doc_id: document id
    :param cid_start: start of CID incrementation
    :param max_len: max sentences per context
    :return:
    """
    # break sentences into chunks
    sent_chunks = [
        ctx['sentences'][i:i + max_len]
        for i in range(0, len(ctx['sentences']), max_len)
    ]

    # create ctx entry for each chunk of sentences
    ctx_chunks = []
    for ctx_offset, sent_chunk in enumerate(sent_chunks):
        cid = cid_start + ctx_offset
        text_start = sent_chunk[0]['start']
        text_end = sent_chunk[-1]['end'] + 1
        if ctx['text'][text_start:text_end].strip():
            new_ctx = {
                'section': ctx['section'],
                'context_id': f"{doc_id}-C{cid:03d}",
                'text': ctx['text'][text_start:text_end],
                'sentences': [{
                    'sentence_id': f"{doc_id}-C{cid:03d}-S{sid:03d}",
                    'start': sent['start'] - text_start,
                    'end': sent['end'] - text_start
                } for sid, sent in enumerate(sent_chunk)]
            }
            ctx_chunks.append(new_ctx)

    return ctx_chunks


def _modify_cids(ctx: Dict, doc_id: str, cid: int) -> Dict:
    """
    Change CID to reflect current CID indexing
    :param ctx:
    :param doc_id:
    :param cid:
    :return:
    """
    new_ctx = {
        'section': ctx['section'],
        'context_id': f"{doc_id}-C{cid:03d}",
        'text': ctx['text'],
        'sentences': [{
            'sentence_id': f"{doc_id}-C{cid:03d}-S{sid:03d}",
            'start': sent['start'],
            'end': sent['end']
        } for sid, sent in enumerate(ctx['sentences'])]
    }
    return new_ctx


def break_long_contexts(data_dict: Dict, max_context_len=15) -> Dict:
    """
    Break any contexts longer than context_len into chunks of size context_len
    :param data_dict:
    :param context_len:
    :return:
    """
    # create new context
    new_contexts = []

    # iterate through old contexts and split if needed
    document_id = data_dict['document_id']
    for context in data_dict['contexts']:
        # skip if no text
        if not context['text']:
            continue
        # break into chunks if len(sentences) > max allowed length
        if len(context['sentences']) > max_context_len:
            context_chunks = _chunkify(context, document_id, len(new_contexts), max_context_len)
            new_contexts += context_chunks
        # or just change context id to match and add
        else:
            new_context = _modify_cids(context, document_id, len(new_contexts))
            new_contexts.append(new_context)

    # create new data dict
    new_data_dict = {k: v for k, v in data_dict.items()}
    new_data_dict['contexts'] = new_contexts
    return new_data_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', help='input directory with processed files')
    parser.add_argument('--output', help='output directory to put new files with split contexts')
    args = parser.parse_args()

    INPUT_DIR = args.input
    OUTPUT_DIR = args.output

    # check input directory
    assert os.path.exists(INPUT_DIR)

    # make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    modified = []
    for fname in tqdm.tqdm(os.listdir(INPUT_DIR)):
        # read input file
        input_file = os.path.join(INPUT_DIR, fname)
        with open(input_file, 'r') as f:
            input_dict = json.load(f)
        # generate new data dict
        output_dict = break_long_contexts(input_dict)
        # record if split
        if len(output_dict['contexts']) > len(input_dict['contexts']):
            modified.append(input_dict['document_id'])
        # write to new file
        output_file = os.path.join(OUTPUT_DIR, fname)
        with open(output_file, 'w') as outf:
            json.dump(output_dict, outf, indent=4)

    # write modified to file
    with open(os.path.join(OUTPUT_DIR, 'info.txt'), 'w') as outf:
        for pid in modified:
            outf.write(f'{pid}\n')

    print('done.')