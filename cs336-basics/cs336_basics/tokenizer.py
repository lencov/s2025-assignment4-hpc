import os
import ast
import time
import heapq
import pickle
import logging
import argparse
import regex as re
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Iterator

from cs336_basics.utils import timed_block

# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)

# Pretokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: List[str] | None = None
):
    if special_tokens is None:
        special_tokens = []

    with timed_block("loading file", logger):
        with open(input_path) as f:
            data = f.read()

    with timed_block("generating pretokens", logger):
        pretokens = defaultdict(int)
        for pretoken in re.finditer(PAT, data, concurrent=True):
            pretokens[pretoken.group(0)] += 1
        del data

    with timed_block("creating vocab", logger):
        vocab = {
            i: token
            for i, token in enumerate(
                [s.encode("utf-8") for s in special_tokens]
                + [bytes([j]) for j in range(256)]
            )
        }

    with timed_block("encoding pretokens", logger):
        pretokens = {
            tuple(bytes((i,)) for i in pretoken.encode("utf-8")): v
            for pretoken, v in pretokens.items()
        }

    with timed_block("creating bytes_2_pretokens index", logger):
        bytes_2_pretokens = defaultdict(set)
        for pretoken in pretokens:
            for byte in pretoken:
                bytes_2_pretokens[byte].add(pretoken)

    with timed_block("calculating pairwise frequencies", logger):
        pairwise_frequencies = defaultdict(int)
        for pretoken in pretokens:
            for i in range(len(pretoken) - 1):
                pair = pretoken[i : i + 2]
                pairwise_frequencies[pair] += pretokens[pretoken]
        pairwise_frequencies_heap = [
            (-freq, token) for token, freq in pairwise_frequencies.items()
        ]
        heapq.heapify(pairwise_frequencies_heap)

    merges = []
    with timed_block("training BPE", logger):
        for _ in tqdm(range(vocab_size - len(vocab))):
            if len(pairwise_frequencies_heap) == 0:
                break

            top_token = None
            top_freq = None
            repush_tokens = []
            while len(pairwise_frequencies_heap) > 0:
                freq, token = heapq.heappop(pairwise_frequencies_heap)

                # Ignore inaccurate frequencies from lazy updates
                if pairwise_frequencies[token] != -freq:
                    continue

                # Update on first iteration
                if top_token is None:
                    top_token = token
                    top_freq = freq
                    continue

                # Break if the frequency is not the same as the top frequency
                if freq != top_freq:
                    heapq.heappush(pairwise_frequencies_heap, (freq, token))
                    break

                # Update top token if lexicographically larger
                if token > top_token:
                    repush_tokens.append(top_token)
                    top_token = token
                else:
                    repush_tokens.append(token)

            for token in repush_tokens:
                heapq.heappush(pairwise_frequencies_heap, (top_freq, token))

            if top_token is None:
                break

            merges.append(top_token)
            top_token_joined = top_token[0] + top_token[1]
            vocab[len(vocab)] = top_token_joined
            changed_keys = []
            for pretoken in (
                bytes_2_pretokens[top_token[0]] & bytes_2_pretokens[top_token[1]]
            ):
                if pretoken not in pretokens:
                    continue
                cur_idx = 0
                pretoken_count = pretokens[pretoken]
                del pretokens[pretoken]
                while cur_idx < len(pretoken) - 1:
                    if pretoken[cur_idx : cur_idx + 2] == top_token:

                        # Decrease count if there's a preceding token before the merge pair
                        if cur_idx > 0:
                            pairwise_frequencies[
                                pretoken[cur_idx - 1 : cur_idx + 1]
                            ] -= pretoken_count
                            changed_keys.append(pretoken[cur_idx - 1 : cur_idx + 1])

                        # Decrease count if there's a next token after the merge pair
                        if cur_idx + 2 < len(pretoken):
                            pairwise_frequencies[
                                pretoken[cur_idx + 1 : cur_idx + 3]
                            ] -= pretoken_count
                            changed_keys.append(pretoken[cur_idx + 1 : cur_idx + 3])

                        pretoken = (
                            *pretoken[:cur_idx],
                            top_token_joined,
                            *pretoken[cur_idx + 2 :],
                        )

                        if cur_idx > 0:
                            pairwise_frequencies[
                                pretoken[cur_idx - 1 : cur_idx + 1]
                            ] += pretoken_count
                            changed_keys.append(pretoken[cur_idx - 1 : cur_idx + 1])
                        if cur_idx + 1 < len(pretoken):
                            pairwise_frequencies[
                                pretoken[cur_idx : cur_idx + 2]
                            ] += pretoken_count
                            changed_keys.append(pretoken[cur_idx : cur_idx + 2])
                    cur_idx += 1
                pretokens[pretoken] = pretoken_count
                for byte in pretoken:
                    bytes_2_pretokens[byte].add(pretoken)
            del pairwise_frequencies[top_token]

            for key in changed_keys:
                heapq.heappush(
                    pairwise_frequencies_heap, (-pairwise_frequencies[key], key)
                )
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ) -> None:
        self._vocab = vocab
        self._inv_vocab = {t: i for i, t in self._vocab.items()}
        self._merges = merges
        self._special_tokens = special_tokens if special_tokens else []

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        return cls(
            pickle.load(open(vocab_filepath, "rb")),
            pickle.load(open(merges_filepath, "rb")),
            special_tokens=special_tokens,
        )

    @classmethod
    def fit(cls, input_path: str, vocab_size: int, special_tokens: List[str]):
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def save(self, path: str = ".", prefix: str = "", overwrite: bool = False):
        start_time = time.time()
        os.makedirs(path, exist_ok=True)
        vocab_path = os.path.join(path, prefix + "vocab.pkl")
        if os.path.exists(vocab_path) and not overwrite:
            raise ValueError(
                f"Vocab file {vocab_path} already exists. Set overwrite flag to true if necessary."
            )
        merges_path = os.path.join(path, prefix + "merges.pkl")
        if os.path.exists(merges_path) and not overwrite:
            raise ValueError(
                f"Merge file {merges_path} already exists. Set overwrite flag to true if necessary."
            )
        with open(vocab_path, "wb+") as f:
            pickle.dump(self._vocab, f)
        with open(merges_path, "wb+") as f:
            pickle.dump(self._merges, f)
        logger.debug(
            "Took %s seconds to save tokenizer state to %s",
            round(time.time() - start_time, 3),
            path,
        )

    @classmethod
    def train_from_file(cls, filepath: str, vocab_size: int, special_tokens: List[str]):
        vocab, merges = train_bpe(filepath, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        ids = []
        pos = 0
        special_regex = re.compile(
            "(" + "|".join(map(re.escape, sorted(self._special_tokens, key=lambda x: len(x), reverse=True))) + ")"
        ) if self._special_tokens else None

        cache = {}
        while pos < len(text):
            if self._special_tokens:
                match = re.search(special_regex, text[pos:])
                if match:
                    start, end = match.span()
                    start += pos
                    end += pos
                else:
                    start, end = len(text), len(text)
            else:
                start, end = len(text), len(text)

            chunk = text[pos:start]
            if chunk:
                for pretoken in re.finditer(PAT, chunk):
                    pretoken = pretoken.group(0)
                    pretoken_original = pretoken
                    if pretoken_original not in cache:
                        pretoken = tuple(bytes((i,)) for i in pretoken.encode("utf-8"))
                        bytes_present = set(pretoken)
                        for pair in self._merges:
                            if pair[0] not in bytes_present or pair[1] not in bytes_present:
                                continue
                            cur_idx = 0
                            while cur_idx < len(pretoken) - 1:
                                if pretoken[cur_idx:cur_idx + 2] == pair:
                                    pretoken = (
                                        *pretoken[:cur_idx],
                                        pair[0] + pair[1],
                                        *pretoken[cur_idx + 2:],
                                    )
                                else:
                                    cur_idx += 1
                            bytes_present.add(pair[0] + pair[1])
                        cache[pretoken_original] = [self._inv_vocab[token] for token in pretoken]
                    ids += cache[pretoken_original]

            if start != end:
                special_token = text[start:end]
                if special_token in self._special_tokens:
                    ids.append(self._inv_vocab[special_token.encode("utf-8")])
            pos = end
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for id in self.encode(s):
                yield id

    def decode(self, ids: List[int]) -> str:
        output = b""
        for id in ids:
            output += self._vocab[id]
        return output.decode("utf-8", errors="replace")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer")
    parser.add_argument("--input_path", type=str, help="Path to the input file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save the tokenizer state"
    )
    parser.add_argument("--vocab_size", type=int, help="Size of the vocabulary")
    parser.add_argument(
        "--special_tokens", type=ast.literal_eval, help="List of special tokens"
    )
    parser.add_argument(
        "--log_level", type=str, default="info", help="Log level (default: info)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite the existing tokenizer state",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())

    tokenizer = Tokenizer.train_from_file(
        args.input_path, args.vocab_size, args.special_tokens
    )
    tokenizer.save(args.output_path, overwrite=args.overwrite)
    logger.info("Tokenizer training completed and saved to %s", args.output_path)
