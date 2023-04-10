import sys 
import re
import sentencepiece as spm
from wenet.utils.file_utils import read_symbol_table
from wenet.dataset.processor import __tokenize_by_bpe_model

def tokenize(origin_list_file,
             symbol_table_file,
             bpe_model_file,
             output_file):
    """ Decode text context list to label list
        Inplace operation

        Args:
            orgin_list_file
            dict
            bpemodel
            file to write

        Returns:
            None
    """
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_file)
    origin_list = open(origin_list_file)
    symbol_table = read_symbol_table(symbol_table_file)
    origin_lines = origin_list.readlines()
    fw = open(output_file,'w')
    for origin_line in origin_lines:
        txt = origin_line.strip()
        parts = [txt]

        label = []
        tokens = []
        for part in parts:
            tokens.extend(__tokenize_by_bpe_model(sp, part))

        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])
        for l in label:
            fw.write(str(l)+" ")
        fw.write('\n')
    fw.close()
    origin_list.close()

if __name__ == "__main__":
    origin_text_file = sys.argv[1]
    symbol_table_file = sys.argv[2]
    bpe_model_file = sys.argv[3]
    fw = sys.argv[4]
    tokenize(origin_text_file,symbol_table_file,bpe_model_file,fw)
