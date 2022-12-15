"""
This code is a modified version of the official documentation for the
transformer library by Hugging Face which can be found below.

We prioritized following the official documentation as close as possible to ensure we're using
robust methods. And also, to improve maintainability as they update the documentation.

https://huggingface.co/transformers/custom_datasets.html#question-answering-with-squad-2-0
"""

from dataclasses import dataclass
import csv
from tqdm import tqdm
import torch
import json
from transformers import DataCollatorWithPadding
from happytransformer.happy_trainer import HappyTrainer, EvalResult
from happytransformer.qa.default_args import ARGS_QA_TRAIN, ARGS_QA_EVAl, ARGS_QA_TEST

@dataclass
class QATrainArgs:
    learning_rate: float = ARGS_QA_TRAIN["learning_rate"]
    num_train_epochs: int = ARGS_QA_TRAIN["num_train_epochs"]
    weight_decay: float = ARGS_QA_TRAIN["weight_decay"]
    adam_beta1: float = ARGS_QA_TRAIN["adam_beta1"]
    adam_beta2: float = ARGS_QA_TRAIN["adam_beta2"]
    adam_epsilon: float = ARGS_QA_TRAIN["adam_epsilon"]
    max_grad_norm:  float = ARGS_QA_TRAIN["max_grad_norm"]
    save_preprocessed_data: bool = ARGS_QA_TRAIN["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_QA_TRAIN["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_QA_TRAIN["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_QA_TRAIN["load_preprocessed_data_path"]
    batch_size: int = ARGS_QA_TRAIN["batch_size"]
    fp16: bool = ARGS_QA_TRAIN["fp16"]


@dataclass
class QAEvalArgs:
    batch_size: int = ARGS_QA_EVAl["batch_size"]
    save_preprocessed_data: bool = ARGS_QA_EVAl["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_QA_EVAl["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_QA_EVAl["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_QA_EVAl["load_preprocessed_data_path"]


@dataclass
class QATestArgs:
    save_preprocessed_data: bool = ARGS_QA_TEST["save_preprocessed_data"]
    save_preprocessed_data_path: str = ARGS_QA_TEST["save_preprocessed_data_path"]
    load_preprocessed_data: bool = ARGS_QA_TEST["load_preprocessed_data"]
    load_preprocessed_data_path: str = ARGS_QA_TEST["load_preprocessed_data_path"]


class QATrainer(HappyTrainer):
    """
    Trainer class for HappyTextClassification
    """

    def train(self, input_filepath, dataclass_args: QATrainArgs):
        """
        See docstring in HappyQuestionAnswering.train()
        """
        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")
        if dataclass_args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")

        self.logger.info("Preprocessing dataset...")
        contexts, questions, answers = self._get_data(input_filepath)
        self.__add_end_idx(contexts, answers)
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)
        self.__add_token_positions(encodings, answers)
        dataset = QuestionAnsweringDataset(encodings)
        data_collator = DataCollatorWithPadding(self.tokenizer)
        self._run_train(dataset, dataclass_args, data_collator)

    def eval(self, input_filepath, dataclass_args: QAEvalArgs):
        """
        See docstring in HappyQuestionAnswering.eval()

        """
        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")
        if dataclass_args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")

        contexts, questions, answers = self._get_data(input_filepath)

        self.__add_end_idx(contexts, answers)
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)
        self.__add_token_positions(encodings, answers)
        eval_dataset = QuestionAnsweringDataset(encodings)
        data_collator = DataCollatorWithPadding(self.tokenizer)

        result = self._run_eval(eval_dataset, data_collator, dataclass_args)
        return EvalResult(loss=result["eval_loss"])


    def test(self, input_filepath, solve, dataclass_args: QATestArgs):
        """
        See docstring in HappyQuestionAnswering.test()

        """

        if dataclass_args.save_preprocessed_data:
            self.logger.info("Saving preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")
        if dataclass_args.load_preprocessed_data:
            self.logger.info("Loading preprocessed data is currently "
                             "not available for question answering models. "
                             "It will be added soon. ")

        contexts, questions = self._get_data(input_filepath, test_data=True)

        return [
            solve(context, question)[0]
            for context, question in
            tqdm(zip(contexts, questions))
        ]

    @staticmethod
    def _get_data(filepath, test_data=False):
        """
        Used to collect
        :param filepath: a string that contains the location of the data
        :return: if test_data = False contexts, questions, answers (all strings)
        else: contexts, questions
        """
        contexts = []
        questions = []
        answers = []
        with open(filepath, newline='', encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contexts.append(row['context'])
                questions.append(row['question'])
                if not test_data:
                    answer = {}
                    answer["answer_text"] = row['answer_text']
                    answer["answer_start"] = int(row['answer_start'])
                    answers.append(answer)
        csv_file.close()

        if not test_data:
            return contexts, questions, answers
        return contexts, questions

    def __add_end_idx(self, contexts, answers):
        for answer, context in zip(answers, contexts):

            gold_text = answer['answer_text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # todo (maybe): strip answer['text'] (remove white space from start and end)
            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2  
            else:
                self.logger.warning(f'End index may not be precise \n'
                                    f'GOLD TEXT: {gold_text}\n'
                                    f'START INDEX: {start_idx}\n'
                                    f'END INDEX: {end_idx}\n\n'
                                    )
                answer['answer_end'] = end_idx

    def __add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


    @staticmethod
    def _generate_json(json_path, input_ids, attention_masks, start_positions, end_positions, name):
        data = {}
        data[name] = []
        data = {
            name: [
                {
                    'input_ids': input_id,
                    'attention_mask': attention_mask,
                    'start_positions': start_positions,
                    'end_positions': end_positions

                }
                for input_id, attention_mask, start_position, end_position in zip(input_ids, attention_masks, start_positions, end_positions)
            ]
        }

        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)


    @staticmethod
    def _get_preprocessed_data(filepath):
        """
        Used for Fetching preprocessed data)
        :param filepath: a string that contains the location of the data
        :return:
        """

        # dataset = load_dataset("csv", data_files={"train": filepath})
        input_ids = []
        attention_mask = []
        start_positions = []
        end_positions = []

        with open(filepath) as json_file:
            data = json.load(json_file)
        json_file.close()

        for case in data["train"]:
            input_ids.append(case['input_ids'])
            attention_mask.append(case['attention_mask'])
            start_positions.append(case['start_positions'])
            end_positions.append(case['end_positions'])

        encodings = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions}

        return encodings


class QuestionAnsweringDataset(torch.utils.data.Dataset):
    """
    A class used to iterate through the training data.
    It used to create  a torch DataLoader object, so that the training data can be
    iterated through in batches easily.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)
