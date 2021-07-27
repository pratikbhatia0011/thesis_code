
import os
import argparse
import json
import nltk
from utils.convert_to_squad_format import get_qad_triples
from utils.utils import get_file_contents
from utils.dataset_utils import read_triviaqa_data, get_question_doc_string

nltk.download('punkt')
sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')


cwd = os.getcwd()


def answer_index_in_document(answer, document):
    answer_list = answer['Aliases'] + answer['NormalizedAliases']
    for answer_string_in_doc in answer_list:
        index = document.find(answer_string_in_doc)
        if index != -1:
            return answer_string_in_doc, index
    return answer['NormalizedValue'], -1


def select_relevant_portion(text):
    paras = text.split('\n')
    selected = []
    done = False
    for para in paras:
        sents = sent_tokenize.tokenize(para)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            for word in words:
                selected.append(word)
                if len(selected) >= 800:
                    done = True
                    break
            if done:
                break
        if done:
            break
        selected.append('\n')
    st = ' '.join(selected).strip()
    return st



def triviaqa_to_squad_format(triviaqa_file, data_dir, output_file):

    triviaqa_json = read_triviaqa_data(triviaqa_file)
    
    qad_triples = get_qad_triples(triviaqa_json)
    
    data = []
    examples_not_found = []
    # example = qad_triples[:100]
    
    for triviaqa_example in qad_triples:
        
        try:
             
            text = get_file_contents(os.path.join(data_dir, triviaqa_example['Filename']), encoding='utf-8')
            context = select_relevant_portion(text)
            
            question_text = triviaqa_example['Question']
            para = {'context': context, 'qas': [{'question': question_text, 'answers': []}]}
            data.append({'paragraphs': [para]})
            qa = para['qas'][0]
            qa['id'] = get_question_doc_string(triviaqa_example['QuestionId'], triviaqa_example['Filename'])
            qa['is_impossible'] = True
            ans_string, index = answer_index_in_document(triviaqa_example['Answer'], context)
        
            if index != -1:
                qa['answers'].append({'text': ans_string, 'answer_start': index})
                qa['is_impossible'] = False
        except OSError:
            examples_not_found.append(triviaqa_example)
    
    triviaqa_as_squad = {'data': data, 'version': '2.0'}
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(triviaqa_as_squad, outfile, indent=2, sort_keys=True, ensure_ascii=False)

    return examples_not_found

examples_not_found = triviaqa_to_squad_format('wikipedia-dev.json', cwd + '/triviaqa-rc/triviaqa-rc/evidence/wikipedia/', 'trivia_as_squad_validation.json')


with open('trivia_as_squad_train.json', encoding='utf-8') as file:
    trivia_train = json.load(file)
