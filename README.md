# BiLex: A Bilingual Lexicon Inducer From Non-Parallel Data #

This software learns a bilingual lexicon from non-parallel data with the help of a small seed lexicon. The technique is described in the following paper:

> Meng Zhang, Haoruo Peng, Yang Liu, Huanbo Luan, and Maosong Sun. Bilingual Lexicon Induction From Non-Parallel Data With Minimal Supervision. In Proceedings of AAAI, 2017.

## Runtime Environment ##

This software has been tested in the following environment, but should work in a compatible one.

- 64-bit Linux
- Python 3.4
- GCC 4.9.4

## Usage ##

1\. Compile the code.

`./compile.sh`

2\. Specify the variables in the `config` file. For example, if `config` contains the following lines:

	config=zh-en
	lang1=zh
	lang2=en

then the data should be located in `data/zh-en` with file extensions `zh` and `en`.

3\. Prepare data according to Step 2. Toy non-parallel data is provided, along with a Chinese-English seed lexicon with 100 word translation pairs. If your seed lexicon has more than 10000 entries, you need to modify the code by redefining `MAX_LEXICON_SIZE`.

4\. Train and obtain the bilingual lexicon.

`./run.sh`

5\. The following files will be generated in `data/zh-en` (the folder specified in `config`):

- word-vec.zh/en: Bilingual word embeddings in a human readable format. From these files vocab.zh/en and vec.zh/en are extracted.
- vocab.zh/en: Vocabularies.
- vec.zh/en: Bilingual word embeddings.
- result: Translations of vocab.zh. For each source word, there will be at most 10 translations after the tab character, each in the format `<translation candidate>:<cosine similarity to the source word vector>`, separated by space and sorted in decreasing order of the cosine similarity. `</s>` is the sentence marker; its translations should be ignored.