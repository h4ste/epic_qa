import functools
import math
import json
import logging
import operator
import heapq
import argparse

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, Set, List, Tuple, Callable, Iterable, Optional, Mapping

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger('epic_eval')


@dataclass
class JudgedSentence:
    """ Represents the nugget annotations for a single sentence. """
    sentence_id: str
    nugget_ids: Set[str] = field(default_factory=set)

    @property
    def sentence_idx(self) -> int:
        return JudgedSentence.id2idx(self.sentence_id)

    @staticmethod
    def id2idx(sentence_id: str) -> int:
        return int(sentence_id.rpartition('-S')[-1])

    @staticmethod
    def get_context_id(sentence_id: str) -> str:
        return sentence_id.rpartition('-')[0]


@dataclass
class JudgedContext:
    """ Holds the sentence annotations for a single context. """
    context_id: str
    sentences: Dict[str, JudgedSentence] = field(default_factory=dict)  # Sentence ID -> Sentence Annotations

    def sentence_idx2id(self, sentence_idx: int) -> str:
        """
        Converts a numeric sentence index to its corresponding ID
        :param sentence_idx: numeric sentence index
        :return: corresponding sentence ID
        """
        return f'{self.context_id}-S{sentence_idx:0>3d}'

    def sentence_for_index(self, sentence_idx: int) -> JudgedSentence:
        """
        Get the annotations for the sentence at the given numeric index
        :param sentence_idx: numeric sentence index
        :return: sentence annotations
        """
        sentence_id = self.sentence_idx2id(sentence_idx)
        return self.sentences.get(sentence_id, JudgedSentence(sentence_id))


@dataclass
class JudgedQuestion:
    question_id: str
    nuggets: Dict[str, str] = field(default_factory=dict)  # Nugget ID -> Nugget Name
    contexts: Dict[str, JudgedContext] = field(default_factory=dict)  # Context ID -> Context Annotations


def load_judgments(path) -> Dict[str, JudgedQuestion]:
    """
    Load judgment json file
    :param path: path to json file (gzipped or extracted)
    :return: Mapping from Question IDs to Question Judgments
    """
    if path.endswith('.gz'):
        # Un-gzip if the path ends in .gz
        import gzip
        with gzip.open(path, 'rb') as in_file:
            judgments = json.load(in_file)
    else:
        # Else assume regular json  file
        if not path.endswith('.json'):
            logger.error('Judgment file %s did not have json extension', path)
        with open(path, 'r') as in_file:
            judgments = json.load(in_file)
    # Dictionary to hold our judgments
    questions: Dict[str, JudgedQuestion] = {}
    for j_question in judgments:
        qid = j_question['question_id']
        assert qid not in questions, f'{qid} encountered twice in judgment file'

        # Create a Question object to hold judgments for this question
        question = questions[qid] = JudgedQuestion(qid)

        # Parse the dictionary of nugget IDs -> nugget names
        question.nuggets = {nugget['nugget_id']: nugget['nugget'] for nugget in j_question['nuggets']}

        # Load sentence-level annotations
        for annotation in j_question['annotations']:
            sentence_id = annotation['sentence_id']
            context_id = JudgedSentence.get_context_id(sentence_id)

            if context_id not in question.contexts:
                question.contexts[context_id] = JudgedContext(context_id)
            context = question.contexts[context_id]

            if sentence_id not in context.sentences:
                context.sentences[sentence_id] = JudgedSentence(sentence_id, nugget_ids=set(annotation['nugget_ids']))
            else:
                context.sentences[sentence_id].nugget_ids.update(annotation['nugget_ids'])
    return questions


@dataclass
class Answer:
    start_sent_id: str
    end_sent_id: str

    def __hash__(self):
        return hash(self.start_sent_id) + 31 * hash(self.end_sent_id)

    def __eq__(self, other):
        return self.start_sent_id == other.start_sent_id and self.end_sent_id == other.end_sent_id

    @property
    def context_id(self) -> str:
        return JudgedSentence.get_context_id(self.start_sent_id)

    @classmethod
    def from_string(cls, s):
        return Answer(*s.split(':', maxsplit=2))


@dataclass
class ScoredAnswer:
    answer: Answer
    gain: float
    nuggets: Set[str] = field(default_factory=set)


@dataclass
class Ranking:
    answers: List[ScoredAnswer] = field(default_factory=list)
    nuggets: Set[str] = field(default_factory=set)
    score: float = 0


def score_answer(answer: Answer,
                 seen_nuggets: Set[str],
                 context: JudgedContext,
                 count_redundant_sentences: bool = False,
                 count_only_novel_nuggets: bool = True,
                 count_filler_sentences: bool = True,
                 merge_novel_sentences: bool = False,
                 ignore_sentence_factor: bool = False):
    """
    Compute the novelty score of an answer given previously seen nuggets and the judgments for the answer's context

    (Novelty Score) = (# of nuggets) * (# of nuggets + 1) / ((# of nuggets) + (sentence factor))

    where (# of nuggets) is restricted to only novel nuggets if count_only_novel_nuggets=True
    and the sentence factor is defined as:
    (sentence factor) =
      ((# of redundant sentences) if count_redundant_sentences=True) +
      ((# of filler sentences) if count_filler_sentences=True) +
      (max((# of novel sentences), 1) if merge_novel_sentences=True,
       (# of novel sentences), otherwise)

    and "redundant sentences" are sentences containing only nuggets retrieved in earlier ranked answers,
    "filler sentences" are sentences with no nuggets at all, and
    "novel sentences" are sentences with novel (i.e., previously unseen) nuggets.


    :param answer: Answer to score
    :param seen_nuggets: Nuggets seen previously in the ranked list of answers
    :param context: Annotated context with sentence-level nugget judgments
    :param count_redundant_sentences: If True, the number of sentences with previously-seen nuggets will be
    added to the sentence factor for scoring, used for Partial scoring
    :param count_only_novel_nuggets: If True, only novel nuggets will contribute to the score, defaults to True
    (setting this to False reduces the measure to NDCG)
    :param count_filler_sentences: If True, sentences without any nuggets will be added to the sentence factor,
    this is included mostly for debugging purposes and should be left as True
    :param merge_novel_sentences: If True, the number of sentences with novel nuggets will be reduced to a maximum
    of 1, with the idea being to not penalize providing multiple sentences as evidence of a new nugget,
    used for Relaxed scoring
    :param ignore_sentence_factor: If True, the sentence factor is ignored and the score is just the number of nuggets
    in the answer, primarily included for debugging purposes
    :return:
    """
    start_idx = JudgedSentence.id2idx(answer.start_sent_id)
    end_idx = JudgedSentence.id2idx(answer.end_sent_id)

    # Count the number of sentences of each type
    num_redundant_sentences = 0
    num_novel_sentences = 0
    num_filler_sentences = 0
    answer_nuggets = set()
    for sent_idx in range(start_idx, end_idx + 1):
        sent = context.sentence_for_index(sent_idx)
        sent_nuggets = sent.nugget_ids
        novel_nuggets = sent_nuggets - seen_nuggets
        if not novel_nuggets:
            if sent_nuggets:
                # We only have redundant nuggets (i.e., nuggets we've seen at earlier ranks)
                num_redundant_sentences += 1
            else:
                # We have no nuggets at all
                num_filler_sentences += 1
        else:
            # We have novel nuggets, yay!
            num_novel_sentences += 1
        # All all nuggets in this sentence to the set of nuggets for the answer
        answer_nuggets.update(sent_nuggets)

    # Sanity check that we have not under- or over-counted any sentences
    num_sentences = end_idx + 1 - start_idx
    assert num_redundant_sentences + num_novel_sentences + num_filler_sentences == num_sentences

    # Determine the sentence factor
    sentence_factor = 0

    if count_filler_sentences:
        sentence_factor += num_filler_sentences

    if count_redundant_sentences:
        sentence_factor += num_redundant_sentences

    if merge_novel_sentences:
        sentence_factor += min(num_novel_sentences, 1)
    else:
        sentence_factor += num_novel_sentences

    if ignore_sentence_factor:
        sentence_factor = 1

    if count_only_novel_nuggets:
        # Remove previously seen nuggets from the set of nuggets in this answer
        # since both datastructures are sets, this does set difference
        answer_nuggets = answer_nuggets - seen_nuggets

    num_nuggets = len(answer_nuggets)

    if num_nuggets == 0:
        # If we neglect to count filter and redundant sentences, and have no novel sentences, we will
        # end up diving by zero if ignore_sentence_factor=False, this isn't really a valid configuration
        # of parameters for scoring, but we return zero here just to be safe
        score = 0.
    else:
        score = num_nuggets * (num_nuggets + 1) / (num_nuggets + sentence_factor)

    return ScoredAnswer(answer, score, answer_nuggets)


def min_max(x: Iterable[int]) -> Tuple[int, int]:
    """
    Jointly compute min and maximum of an iterable in a single pass

    :param x: Iterable of integers
    :return: Tuple of (min, max) integers in x
    """
    maximum = -math.inf
    minimum = math.inf
    for i in x:
        if i > maximum:
            maximum = i
        if i < minimum:
            minimum = i
    return minimum, maximum


def get_potential_answers(context: JudgedContext) -> Iterable[Answer]:
    """
    Generate potential answers for a judged context. Note: we skip any answers that begin before the first annotated
    sentence or continue beyond the last annotated sentence in the context, as those answers will always be worse than
    answers that begin with the first or end with the last annotated sentence.

    TODO: We could prune any answers that don't start and end on an annotated sentence to speed this up

    :param context: JudgedContext
    :return: Iterable of *most* potential answers in that context
    """
    indices = map(operator.attrgetter('sentence_idx'), context.sentences.values())
    min_idx, max_idx = min_max(indices)

    logger.debug('Generating %d potential answers for context %s',
                 (max_idx + 1 - min_idx) * (max_idx + 2 - min_idx) / 2,
                 context.context_id)
    for i in range(min_idx, max_idx + 1):
        sent_i_id = context.sentence_idx2id(i)
        for j in range(i, max_idx + 1):
            yield Answer(sent_i_id, context.sentence_idx2id(j))


def get_ideal_ranking(question: JudgedQuestion,
                      answers: List[Answer],
                      score_fn: Callable[[Answer, Set[str], JudgedContext], ScoredAnswer] = score_answer,
                      max_len: int = 1000,
                      k: int = 10) -> Ranking:
    """
    Perform a beam-search over candidate answers to produce an "ideal" ranking of answers to the given question
    based on its judgments and the given scoring function
    :param question: Judgments for a single question
    :param answers: Candidate answers for that question (i.e., from  `get_potential_answers`)
    :param score_fn: scoring function (used to compute the gain of an answer)
    :param max_len: maximum ranking length
    :param k: beam-width
    :return: Optimal ranking for the question based on the given score_fn and judgments
    """

    # Candidate rankings in the beam, initialize with a single empty ranking
    rankings = [Ranking()]
    for r in range(max_len):
        # DCG denominator is log2(r + 1) where r starts at 1, so we need to add 2 since our r starts at 0
        dcg_denom = math.log2(r + 2)
        # Store all potential rankings up to depth r for all candidates in the beam
        candidates = list()
        # Iterate over all candidate rankings in the beam
        for ranking in rankings:
            # Keep track of answers we've seen so we don't include duplicates in our ideal ranking
            seen_answers = frozenset(map(operator.attrgetter('answer'), ranking.answers))
            # Iterate over all potential answers
            for a, answer in enumerate(answers):
                if answer not in seen_answers:
                    # Calculate the score of this answer given the nuggets we've seen in this candidate ranking
                    scored_answer = score_fn(answer, ranking.nuggets, question.contexts[answer.context_id])
                    # Prune answers with zero scores
                    if scored_answer.gain > 0:
                        # Calculate discounted score for the ranking obtained by adding this answer to the
                        # current candidate
                        dcg = ranking.score + (scored_answer.gain / dcg_denom)
                        candidates.append((ranking, scored_answer, dcg))
        # If none of the answers increase the score for any ranking in the beam, we stop early
        if not candidates:
            logger.debug('Exhausted all answers by rank %d', r + 1)
            break
        # Take the top K scoring rankings from all candidates
        top_k = heapq.nlargest(k, candidates, key=operator.itemgetter(2))
        # Update our current beam candidates!
        rankings = []
        for p, a, dcg in top_k:
            rankings.append(Ranking(answers=p.answers + [a],  # Add the best answer to its candidate ranking
                                    nuggets=p.nuggets.union(a.nuggets),  # Update the nuggets in the new ranking
                                    score=dcg))

    # Return the best ranking from our beam
    return max(rankings, key=operator.attrgetter('score'))


@dataclass
class Submission:
    runtag: str
    rankings: Dict[str, List[Answer]]  # Question ID -> Ranked list of Answers


def load_submission(path, max_len=1000):
    """
    Load a submission file
    :param path: path to submission file
    :param max_len: maximum retrieval length
    :return:
    """
    parsed_rankings = defaultdict(list)
    with open(path, 'r') as in_file:
        for line in in_file:
            qid, q0, answer, rank, score, runtag = line.split(maxsplit=5)
            parsed_rankings[qid].append([Answer.from_string(answer), rank, score])

    rankings = {}
    for qid, parsed_ranking in parsed_rankings.items():
        ranking = sorted(parsed_ranking, key=operator.itemgetter(2), reverse=True)
        ranking_len = len(ranking)
        if ranking_len < max_len:
            logger.debug('Query %s had only %d answers (maximum is %d)', qid, ranking_len, max_len)
        if ranking_len > max_len:
            logger.error('Query %s had %d answers (maximum is %d), top-%d answers (by score) will be evaluated',
                         qid, ranking_len, max_len, ranking_len)
            ranking = ranking[:max_len]
        rankings[qid] = [answer for answer, rank, score in ranking]

    return Submission(runtag=runtag.strip(), rankings=rankings)


def calc_discounted_score(ranking: List[Answer],
                          judgments: JudgedQuestion,
                          score_fn: Callable[
                              [Answer, Set[str], JudgedContext], ScoredAnswer] = score_answer) -> Ranking:
    """
    Calculate the discounted score where the discount is logarithmic with respect to the rank of the document.
    This is essentially Discounted Cumulative Gain from NDCG but using our score_fn instead of the gain
    :param ranking: Ranking of answers for a single question (i.e., the ranking to score)
    :param judgments: Judgments for this question
    :param score_fn: scoring function to compute the gain
    :return: Ranking (with individual gain scores, nuggets, and discounted score)
    """
    dcg = 0
    answers = []
    nuggets = set()
    for r, answer in enumerate(ranking, start=1):
        context_id = answer.context_id
        if context_id in judgments.contexts:
            answer = score_fn(answer, nuggets, judgments.contexts[context_id])
            answers.append(answer)
            nuggets.update(answer.nuggets)
            dcg += answer.gain / math.log2(r + 1)
    return Ranking(answers=answers, nuggets=nuggets, score=dcg)


def load_ranking_file(ideal_ranking_file: str) -> Dict[str, Dict[str, float]]:
    """
    Load a ranking file (used for ideal ranking scores)
    :param ideal_ranking_file: path to the ranking file to read
    :return: Mapping from query ID -> metric name -> score
    """
    ideal_scores = defaultdict(dict)
    with open(ideal_ranking_file, 'r') as in_file:
        for line in in_file:
            ideal, qid, metric, score = line.split(maxsplit=3)
            assert ideal == 'ideal', 'ideal ranking list had invalid name: expected \'ideal\''
            ideal_scores[qid][metric] = float(score.strip())
    return ideal_scores


def save_ranking_file(ideal_ranking_file: str,
                      ideal_scores: Mapping[str, Mapping[str, float]]) -> None:
    """
    Save a ranking file (used for ideal ranking scores)
    :param ideal_ranking_file: path to save ranking file
    :param ideal_scores: Mapping from query ID -> metric name -> score
    :return: None
    """

    with open(ideal_ranking_file, 'w') as out_file:
        for qid, metrics in ideal_scores.items():
            for metric, score in metrics.items():
                out_file.write(f'ideal\t{qid}\t{metric}\t{score}\n')


# noinspection PyTypeChecker
score_fns: Mapping[str, Callable[[Answer, Set[str], JudgedContext], ScoredAnswer]] = {
    'NDNS-Partial': functools.partial(score_answer,
                                      merge_novel_sentences=True),
    'NDNS-Relaxed': functools.partial(score_answer,
                                      count_redundant_sentences=True,
                                      merge_novel_sentences=True),
    'NDNS-Exact': functools.partial(score_answer,
                                    count_redundant_sentences=True),
    'NDCG-Exact': functools.partial(score_answer,
                                    count_redundant_sentences=True,
                                    count_only_novel_nuggets=False),
    'NDCG': functools.partial(score_answer,
                              count_only_novel_nuggets=False,
                              ignore_sentence_factor=True)
}


def main(judgment_file: str,
         submission_file: str,
         task: str,
         metrics: List[str],
         ideal_ranking_file: Optional[str] = None):
    judgments = load_judgments(judgment_file)
    submission = load_submission(submission_file)
    runtag = submission.runtag

    # Filter judgments and sanity check submissions based on the given task
    if task == 'expert':
        for qid in list(judgments.keys()):
            if not qid.startswith('E'):
                del judgments[qid]
        for qid in list(submission.rankings.keys()):
            if not qid.startswith('E'):
                logger.warning('--task=expert but submission file contains consumer query %s', qid)
                del submission.rankings[qid]
    elif task == 'consumer':
        for qid in list(judgments.keys()):
            if not qid.startswith('C'):
                del judgments[qid]
        for qid in list(submission.rankings.keys()):
            if not qid.startswith('C'):
                logger.warning('--task=consumer but submission file contains expert query %s', qid)
                del submission.rankings[qid]
    else:
        assert task == 'all', f'encountered unexpected task: {task} should be one of {{expert, consumer, all}}'

    # If we don't have an ideal ranking file, assume the default path
    if not ideal_ranking_file:
        ideal_ranking_file = f'.{task}_ideal_ranking_scores.tsv'

    # Try to load ideal ranking file, otherwise set it to empty
    save_ideal_scores = False
    try:
        ideal_scores = load_ranking_file(ideal_ranking_file)
    except IOError:
        logger.info('--ideal-ranking-file not found; ideal rankings will be calculated and '
                    'saved to %s for future use (Note: this process can be slow)', ideal_ranking_file)
        save_ideal_scores = True
        ideal_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    # Store scores as Question ID -> Metric -> Score
    scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for qid, judged_question in judgments.items():
        if qid not in submission.rankings:
            logger.error('Submission had no answers for question %s', qid)
            scores[qid] = {metric: 0. for metric in metrics}
        else:
            # Used to cache candidate answers for this question in the event we need to compute ideal scores
            answers: List[Answer] = []
            for metric in metrics:
                score_fn = score_fns[metric]

                # If we have an ideal score already, we can just use it!
                if metric in ideal_scores[qid]:
                    ideal_score = ideal_scores[qid][metric]
                else:
                    # If we don't have an ideal score already, we need to compute it
                    save_ideal_scores = True
                    if not answers:
                        # Get all candidate answers to the question
                        for context in judged_question.contexts.values():
                            answers.extend(get_potential_answers(context))
                        logger.debug('Generated %d candidate answers for question %s',
                                     len(answers),
                                     judged_question.question_id)

                    # Calculate ideal ranking
                    ideal_ranking = get_ideal_ranking(judged_question, answers=answers, score_fn=score_fn,
                                                      k=1 if metric.startswith('NDCG') else 10)
                    # Save the score of the ideal ranking for this metric for this question
                    ideal_score = ideal_scores[qid][metric] = ideal_ranking.score

                # Calculate discounted cumulative score for this ranking
                submission_ranking = calc_discounted_score(submission.rankings[qid], judged_question, score_fn=score_fn)
                submission_score = submission_ranking.score

                # Normalize by ideal score
                score = submission_score / ideal_score
                scores[qid][metric] = score

    if save_ideal_scores:
        logger.debug('Saving ideal ranking scores to %s', ideal_ranking_file)
        save_ranking_file(ideal_ranking_file, ideal_scores)

    for qid, metrics in scores.items():
        for metric, score in metrics.items():
            print(runtag, qid, metric, score, sep='\t')
    for metric in metrics:
        print(runtag, 'MEAN', metric, mean([metrics[metric] for metrics in scores.values()]), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('judgment_file', help='judgment file')
    parser.add_argument('submission_file', help='submission file')
    parser.add_argument('ideal_ranking_file', nargs='?', help='ideal ranking file')
    parser.add_argument('--task', choices=['expert', 'consumer', 'all'])
    parser.add_argument('--metrics', choices=list(score_fns.keys()),
                        default=['NDNS-Partial', 'NDNS-Relaxed', 'NDNS-Exact'],
                        help='metrics to report')

    args = parser.parse_args()
    main(**vars(args))
