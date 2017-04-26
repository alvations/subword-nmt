#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
from collections import defaultdict, Counter

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser

def get_vocabulary(fobj, is_dict=False):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    for line in fobj:
        if is_dict:
            word, count = line.strip().split()
            vocab[word] = int(count)
        else:
            for word in line.split():
                vocab[word] += 1
    return vocab

def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+second
    for j, word, old_word, freq in changed:

        # Find all instances of pair in the old_word, and update frequency/indices around it
        i = 0
        # Keep moving down the old_word string until we cannot find 
        # the first char in the new_pair.
        while True: 
            try:
                # Find the next occurence of the first character in the new_pair.
                i = old_word.index(first, i)
            except ValueError:
                break
            # Checks that old_word[i:i+1] is the same as new_pair.
            # (i) `i < len(old_word)-1` checks that the index i is not the last character.
            # (ii) `old_word[i+1]` checks that the char after the index is the second char in the new_pair.
            if i < len(old_word)-1 and old_word[i+1] == second:
                # `if i` checks that i is non-zero. 
                # We can skip the first char since there's no previous bigram.
                if i:
                    # Find the previous bigram and reduce its count. 
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                # `if < len(old_word)-2` checks that the new_pair is not at the end of the old_word.
                if i < len(old_word)-2:
                    # The multiple if conditions that follows checks that the bigram after i and i+1 
                    # is not the same as new_pair to avoid double-counting consecutive pairs.
                    # `old_word[i+2] != first` checks that two chars after i, it isn't the same as 
                    #                          the first char in the new_pair.
                    # `old_word[i+3] != second` checks that three chars after i, it isn't the same 
                    #                           as the second char in the new_pair.
                    # `i >= len(old_word)-3` checks that the i index is one of the last 4 chars in old_word.
                    # @rico: Is the `i >= len(old_word)-3` check to avoid IndexError?
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        # Find the next bigram and reduce its count.
                        # `nex` is the next bigram after new_pair.
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                # Now we move the ith index to two chars to the right when 
                # old_word[i:i+1] is the same as new_pair.
                i += 2
            else: # Otherwise, we move one char to the right.
                i += 1

        # Find all instances of pair in the new *word*, and update frequency/indices around it
        # Reset the index to the start of the string.
        i = 0
        # Similarly, we keep moving down the new *word* string until we cannot find 
        # the first char in the new_pair.
        while True:
            try:
                i = word.index(new_pair, i)
            except ValueError:
                break
            # We are sure that the new_pair is in the new *word* so there's no need to
            # do an outer check as what was done in the old_word.
            if i: # `if i` checks that i is non-zero, skip the first char since there's no previous bigram.
                prev = word[i-1:i+1]
                # This time, we add the frequency back to the statistics and indices.
                stats[prev] += freq
                indices[prev][j] += 1
            # The multiple if conditions that follows checks that the bigram after i and i+1 
            # is not the same as new_pair to avoid double-counting consecutive pairs.
            # `i < len(word)-1` checks if i is not the last character.
            # `word[i+1]` checks that the next char is not the new_pair.
            if i < len(word)-1 and word[i+1] != new_pair:
                # `nex` is the next bigram after new_pair.
                nex = word[i:i+2]
                # We add the frequency back to the statistics and indices.
                stats[nex] += freq
                indices[nex][j] += 1
            # We move one char down the new *word*
            i += 1


def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split())

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes

def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def main(infile, outfile, num_symbols, min_frequency=2, verbose=False, is_dict=False):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    outfile.write('#version: 0.2\n')

    vocab = get_vocabulary(infile, is_dict)
    vocab = dict([(tuple(x[:-1])+(x[-1]+'</w>',) ,y) for (x,y) in vocab.items()])
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)
    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(stats.values()) / 10
    for i in range(num_symbols):
        if stats:
            most_frequent = max(stats, key=lambda x: (stats[x], x))

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = stats[most_frequent] * i/(i+10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent] < min_frequency:
            sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        outfile.write('{0} {1}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)
        stats[most_frequent] = 0
        if not i % 100:
            prune_stats(stats, big_stats, threshold)


if __name__ == '__main__':

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    main(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input)
