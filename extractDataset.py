import os

import torch
import logging

from igraph import Graph, plot

from enum import Enum

from tqdm import tqdm

from pathlib import Path

# DATA HANDLING CLASSES
from treelstm import Vocab
# IMPORT CONSTANTS
from treelstm import Constants
# CONFIG PARSER
from config import parse_args
# UTILITY FUNCTIONS
from treelstm import utils
# DATASET CLASS FOR SICK DATASET
from treelstm import SICKConstitencyDataset
from treelstm import SICKDataset as SICKDependencyDataset
from treelstm import TreeType
import matplotlib.pyplot as plt

class DatasetType(Enum):
    TRAIN = 1
    DEV = 2
    TEST = 3


class TreeSaver:

    def __init__(self, folder_name, treetype):
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        self.folder_name = folder_name
        self.vocab = []
        self.labels = []
        self.tree = []
        self.input = []
        self.treetype = treetype

    def remember(self, dataset):
        vocab_index = 0
        for idx, label in dataset.vocab.idxToLabel.items():
            assert vocab_index == idx
            self.vocab.append(label)
            vocab_index += 1

        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Saving unique trees'):
            ltree, linput, rtree, rinput, label = dataset[indices[idx]]
            if self.treetype == TreeType.DEPENDENCY:
                assert ltree.size() == len(linput)
                assert rtree.size() == len(rinput)
            else:
                assert ltree.leaf_size() == len(linput)
                assert rtree.leaf_size() == len(rinput)
            self.tree.append(ltree)
            self.input.append(linput.tolist())
            self.tree.append(rtree)
            self.input.append(rinput.tolist())
            self.construct_label_tree(linput, dataset.vocab)
            self.construct_label_tree(rinput, dataset.vocab)

    def construct_label_tree(self, input, vocab):
        line = []
        for inp in input:
            line.append(vocab.getLabel(inp.tolist()))
        self.labels.append(line)

    def save_in_file(self, list_to_save, output_file):
        with open(os.path.join(self.folder_name, output_file), "w") as file:
            for element in list_to_save:
                file.write("{}\n".format(element))

    def save(self, vocab_file, tree_file, input_file, label_file, build_graph):
        self.save_in_file(self.vocab, vocab_file)
        self.save_in_file(self.tree, tree_file)
        self.save_in_file(self.input, input_file)
        self.save_in_file(self.labels, label_file)
        if build_graph:
            for tree in self.tree:
                self.save_graph(tree)

    def save_graph(self, tree):
        if self.treetype == TreeType.DEPENDENCY:
            gx = Graph()
            tree.construct_graph(gx.add_vertices, gx.add_edge)
            print(gx.get_edgelist())
            lay = gx.layout_reingold_tilford()
            plot(gx, layout=lay)
            # plt.show()


class Helper:

    def __init__(self, logger, train_dir, dev_dir, test_dir):
        self.logger = logger
        self.train_dir = train_dir
        self.dev_dir = dev_dir
        self.test_dir = test_dir

    def get_vocab(self):

        if hasattr(self, '_vocab'):
            return self._vocab

        # write unique words from all token files
        sick_vocab_file = os.path.join(args.data, 'sick.vocab')
        if not os.path.isfile(sick_vocab_file):
            token_files_b = [os.path.join(split, 'b.toks') for split in [self.train_dir, self.dev_dir, self.test_dir]]
            token_files_a = [os.path.join(split, 'a.toks') for split in [self.train_dir, self.dev_dir, self.test_dir]]
            token_files = token_files_a + token_files_b
            sick_vocab_file = os.path.join(args.data, 'sick.vocab')
            utils.build_vocab(token_files, sick_vocab_file)

        # get vocab object from vocab file previously written
        self._vocab = Vocab(filename=sick_vocab_file,
                            data=[Constants.PAD_WORD, Constants.UNK_WORD,
                                  Constants.BOS_WORD, Constants.EOS_WORD])
        self.logger.debug('==> SICK vocabulary size : %d ' % self._vocab.size())

        return self._vocab

    def get_dataset(self, dataset_type, tree_type):
        if tree_type == TreeType.DEPENDENCY:
            file_name = 'dependency'
        elif tree_type == TreeType.CONSTITUENCY:
            file_name = 'constituency'
        if dataset_type == DatasetType.TRAIN:
            file_name = '{}_train.pth'.format(file_name)
            dir = self.train_dir
            type = 'train'
        elif dataset_type == DatasetType.DEV:
            file_name = '{}_dev.pth'.format(file_name)
            dir = self.dev_dir
            type = 'dev'
        elif dataset_type == DatasetType.TEST:
            file_name = '{}_test.pth'.format(file_name)
            dir = self.test_dir
            type = 'test'
        else:
            assert False

        dataset_file = os.path.join(args.data, file_name)
        if os.path.isfile(dataset_file) and False:
            dataset = torch.load(dataset_file)
        else:
            if tree_type == TreeType.DEPENDENCY:
                dataset = SICKDependencyDataset(dir, self.get_vocab(), args.num_classes)
            else:
                dataset = SICKConstitencyDataset(dir, self.get_vocab(), args.num_classes)
            torch.save(dataset, dataset_file)
        self.logger.debug('==> Size of %s data   : %d ' % (type, len(dataset)))

        return dataset


def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname) + '.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    dependency_dir = os.path.join(args.extract_dir, 'dependency')
    constituency_dir = os.path.join(args.extract_dir, 'constituency')

    helper = Helper(logger, train_dir, dev_dir, test_dir)
    # vocab = helper.get_vocab()

    dependency_saver = TreeSaver(dependency_dir, TreeType.DEPENDENCY)
    dependency_saver.remember(helper.get_dataset(DatasetType.TRAIN, TreeType.DEPENDENCY))
    dependency_saver.remember(helper.get_dataset(DatasetType.DEV, TreeType.DEPENDENCY))
    dependency_saver.remember(helper.get_dataset(DatasetType.TEST, TreeType.DEPENDENCY))
    dependency_saver.save('vocab.txt', 'tree.txt', 'input.txt', 'sentences.txt', True)

    constituency_saver = TreeSaver(constituency_dir, TreeType.CONSTITUENCY)
    constituency_saver.remember(helper.get_dataset(DatasetType.TRAIN, TreeType.CONSTITUENCY))
    constituency_saver.remember(helper.get_dataset(DatasetType.DEV, TreeType.CONSTITUENCY))
    constituency_saver.remember(helper.get_dataset(DatasetType.TEST, TreeType.CONSTITUENCY))
    constituency_saver.save('vocab.txt', 'tree.txt', 'input.txt', 'sentences.txt', True)


if __name__ == '__main__':
    main()
