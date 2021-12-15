import argparse
import pathlib
from dataclasses import dataclass, field
from typing import Set, List
from xml.etree import ElementTree as ET
from pathlib import Path

import praw

from tqdm.auto import tqdm


# @dataclass
# class SectionTree:
#     title: str
#     children: List = field(default_factory=list)


@dataclass
class SectionNode:
    text: str
    title: str
    children: List = field(default_factory=list)


def extract_as_xml(submission):
    doc = ET.Element('doc', {'id': submission.id, 'url': submission.url})

    title = ET.SubElement(doc, 'title')
    title.text = submission.title

    text = ET.SubElement(doc, 'text', {'xml:splace': 'preserve'})
    text.text = submission.selftext

    root = SectionNode('[root]', 'root')
    submission.comment_sort = "top"
    submission.comment_limit = 30
    comment_queue = [(root, [], c) for c in submission.comments]
    while comment_queue:
        parent, authors, comment = comment_queue.pop(0)
        if isinstance(comment, praw.models.MoreComments) or comment.score <= 0 or comment.body == '[removed]' or comment.body == '[deleted]':
            continue
        else:
            author = comment.author.name if comment.author else '[deleted]'
            node = SectionNode(
                text=comment.body,
                title='by:' + author + ''.join(f' re:{a}' for a in authors[::-1]),
            )
            parent.children.append(node)
            # print('adding', node.title, 'to', parent.title)
            comment_queue.extend([(node, authors + [author], reply) for reply in comment.replies])

    def add_comment(sections_tag: ET.Element, comment: SectionNode):
        subsection = ET.SubElement(sections_tag, 'section')
        subsection_title = ET.SubElement(subsection, 'title')
        subsection_title.text = comment.title
        subsection_text = ET.SubElement(subsection, 'text', {'xml:space': 'preserve'})
        subsection_text.text = comment.text
        if comment.children:
            subsection_sections = ET.SubElement(subsection, 'sections')
            for subcomment in comment.children:
                add_comment(subsection_sections, subcomment)

    sections = ET.SubElement(doc, 'sections')
    for comment in root.children:
        add_comment(sections, comment)

    #         subsection = ET.SubElement(subsections, 'section')
    #         substitle = ET.SubElement(subsection, 'title')
    #         second_author_name = second_comment.author.name if second_comment.author else 'None'
    #         substitle.text = f'{second_author_name} re:{top_author_name}'
    #         substext = ET.SubElement(subsection, 'text', {'xml:space': 'preserve'})
    #         substext.text = second_comment.body

    # submission.comment_sort = "top"
    # submission.comment_limit = 10
    # for top_comment in submission.comments:
    #     if isinstance(top_comment, praw.models.MoreComments) or top_comment.score <= 0:
    #         break
    #     if top_comment.body == '[removed]':
    #         continue
    #     section = ET.SubElement(sections, 'section')
    #     stitle = ET.SubElement(section, 'title')
    #     top_author_name = top_comment.author.name if top_comment.author else 'None'
    #     stitle.text = top_author_name
    #     stext = ET.SubElement(section, 'text', {'xml:space': 'preserve'})
    #     stext.text = top_comment.body
    #
    #     top_comment.reply_sort = 'top'
    #     top_comment.reply_limit = 10
    #     secondary_comments = top_comment.refresh().replies
    #     subsections = ET.SubElement(section, 'sections')
    #     delete_subsections = True
    #     for second_comment in secondary_comments:
    #         if isinstance(second_comment, praw.models.MoreComments) or second_comment.score <= 0:
    #             break
    #         if second_comment.body == '[removed]':
    #             continue
    #         delete_subsections = False
    #         subsection = ET.SubElement(subsections, 'section')
    #         substitle = ET.SubElement(subsection, 'title')
    #         second_author_name = second_comment.author.name if second_comment.author else 'None'
    #         substitle.text = f'{second_author_name} re:{top_author_name}'
    #         substext = ET.SubElement(subsection, 'text', {'xml:space': 'preserve'})
    #         substext.text = second_comment.body
    #     if delete_subsections:
    #         section.remove(subsections)

    return ET.ElementTree(doc)


def extract_to_dir(out_dir, submissions):
    if not isinstance(out_dir, pathlib.Path):
        out_dir = Path(out_dir)
    for i, submission in enumerate(tqdm(submissions)):
        sid = submission.id
        out_file = out_dir / f'{sid}.xml'
        if out_file.is_file():
            tqdm.write(f'Skipping existing submission {sid}')
        else:
            doc = extract_as_xml(submission)
            doc.write(out_file)
            tqdm.write(f'Saved {sid} to {out_file}')


def main(subreddit: str, query: str, out_dir: str):
    reddit = praw.Reddit(
        client_id='O7HBFkqXVGW-FA',
        client_secret='b39T3bUPg85DQgqAxV7nKrRMk8EOpw',
        user_agent='idfk'
    )
    subreddit = reddit.subreddit(subreddit)
    submissions = subreddit.search(query, syntax='lucene', sort='top', limit=None)
    extract_to_dir(out_dir, submissions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subreddit', nargs='?', type=str, default='askscience',
                        help='name of subreddit to crawl, e.g., askscience')
    parser.add_argument('--query', nargs='?', type=str, default='flair:COVID-19',
                        help='lucene query parameters, e.g., flair:COVID-19')
    parser.add_argument('--output-dir', required=True, type=pathlib.Path,
                        help='output directory')
    args = parser.parse_args()
    main(subreddit=args.subreddit,
         query=args.query,
         out_dir=args.output_dir)
