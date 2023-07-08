import logging
from transformers import AutoTokenizer, TrainingArguments, Trainer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, AutoModel
import torch
import tempfile
import math
from datasets import load_dataset, load_from_disk, DatasetDict
from happytransformer.args import TrainArgs
from happytransformer.fine_tuning_util import EvalResult
from typing import Union

class HappyTransformer():

    def __init__(self, model_type, model_name, model_class: AutoModel, load_path="", use_auth_token: Union[str, bool] = None):

        self.logger = logging.getLogger(__name__)

        self._model_class = model_class

        if load_path != "":
            self._init_model(model_type, load_path, use_auth_token)
        else:
            self._init_model(model_type, model_name, use_auth_token)


        handler = logging.StreamHandler()
        handler.addFilter(logging.Filter('happytransformer'))
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[handler]
        )

        self.device = None

        if torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                self.device = torch.device("mps")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        if not self.device:
            self.device = torch.device("cpu")

        if self.device.type != 'cpu':
            self.model.to(self.device)

        self.logger.info("Using model: %s", self.device)

        # Set within the child classes.
        self._data_collator = None
        self._t_data_file_type = None
        self._type = None

    def train(self, input_filepath: str ,  args: TrainArgs, eval_filepath: str = "", ):
        if type(args) == dict:
            raise ValueError("Dictionary training arguments are no longer supported as of Happy Transformer version 2.5.0.")

        train_tok_data, eval_tok_data = self._preprocess_data_train(input_filepath=input_filepath,
                                                              eval_filepath=eval_filepath,
                                                              args=args)

        self._run_train(train_tok_data, eval_tok_data, args,  self._data_collator)



    def eval(self, input_filepath, args):
        if type(args) == dict:
            raise ValueError(
                "Dictionary evaluaging arguments are no longer supported as of Happy Transformer version 2.5.0.")


        tokenized_dataset = self._preprocess_data_eval(input_filepath, args)

        result = self._run_eval(tokenized_dataset, self._data_collator, args)

        return EvalResult(loss=result["eval_loss"])

    def test(self, input_filepath, args):
        raise NotImplementedError()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _preprocess_data_train(self, input_filepath, eval_filepath, args: TrainArgs):

        if not args.load_preprocessed_data:
            if eval_filepath == "":
                all_raw_data = load_dataset(self._t_data_file_type, data_files={"train": input_filepath}, split="train")
                all_raw_data = all_raw_data.shuffle(seed=42)
                split_text_data = all_raw_data.train_test_split(test_size=args.eval_ratio)
                train_tok_data = self._tok_function(split_text_data["train"], args)
                eval_tok_data = self._tok_function(split_text_data["test"], args)
            else:
                raw_data = load_dataset(self._t_data_file_type, data_files={"train": input_filepath, "eval": eval_filepath})
                train_tok_data = self._tok_function(raw_data["train"], args)
                eval_tok_data = self._tok_function( raw_data["eval"], args)
        else:
            if args.save_preprocessed_data_path.endswith(".json"):
                raise ValueError(
                    "As of version 2.5.0 preprocessed files are not longer saved as json files. Please preprocess your data again")

            self.logger.info("Loading dataset from %s...", args.load_preprocessed_data_path)
            tok_data = load_from_disk(args.load_preprocessed_data_path)
            train_tok_data = tok_data["train"]
            eval_tok_data = tok_data["eval"]

        if args.save_preprocessed_data:

            if args.load_preprocessed_data:
                self.logger.warning("Both save_preprocessed_data and load_data are enabled,")

            if args.save_preprocessed_data_path.endswith(".json"):
                raise ValueError(
                    "As of version 2.5.0 preprocessed files are not longer saved as json files. Please provide a path to a folder.")


            combined_tok = DatasetDict({"train": train_tok_data, "eval": eval_tok_data})
            combined_tok.save_to_disk(args.save_preprocessed_data_path)

        return train_tok_data, eval_tok_data


    def _preprocess_data_eval(self, input_filepath, args: TrainArgs):
        if not args.load_preprocessed_data:
            self.logger.info("Preprocessing dataset...")
            datasets = load_dataset(self._t_data_file_type, data_files={"eval": input_filepath})
            tokenized_dataset = self._tok_function(datasets["eval"], args)

        else:
            self.logger.info("Loading dataset from %s...", args.load_preprocessed_data_path)
            tokenized_dataset = load_from_disk(args.load_preprocessed_data_path +"/eval")

        if args.save_preprocessed_data:
            if args.load_preprocessed_data:
                self.logger.warning("Both save_preprocessed_data and load_data are enabled.")

            self.logger.info("Saving evaluating dataset to %s...", args.save_preprocessed_data_path)
            save_dataset = DatasetDict({"eval": tokenized_dataset})
            save_dataset.save_to_disk(args.save_preprocessed_data_path)

        return tokenized_dataset

    def _tok_function(self, raw_dataset, args: TrainArgs):
        raise NotImplementedError()

    def _get_training_args(self, args):
        if self.device.type != "cuda":
            if args.fp16:
                ValueError("fp16 is only available when CUDA/ a GPU is being used. ")

        if self._type == "tt":
            arg_class = Seq2SeqTrainingArguments
        else:
            arg_class = TrainingArguments

        return arg_class(
            deepspeed=None if args.deepspeed == "" else args.deepspeed,
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.num_train_epochs,
            report_to=["none"] if len(args.report_to) == 0 else args.report_to,
            save_strategy="steps" if args.save_steps > 0 else "no",
            save_steps=args.save_steps,
            evaluation_strategy="steps" if args.eval_steps > 0 else "no",
            eval_steps=args.eval_steps,
            logging_strategy="steps" if args.logging_steps > 0 else "no",
            logging_steps = args.logging_steps,
            per_device_train_batch_size=args.batch_size,
            fp16=args.fp16,
            gradient_accumulation_steps=args.gas,
            use_mps_device= True if self.device.type == "mps" else False
        )


    def _run_train(self, train_dataset, eval_dataset, args, data_collator):
        """
        :param dataset: a child of torch.utils.data.Dataset
        :param args: a dataclass that contains settings
        :return: None
        """
        training_args = self._get_training_args(args)

        if self._type == "tt":
            train_class = Seq2SeqTrainer
        else:
            train_class = Trainer

        trainer = train_class(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

    def _run_eval(self, dataset, data_collator, args):
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            eval_args = self._get_eval_args(tmp_dir_name, args)
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=dataset,
                data_collator=data_collator
            )
            return trainer.evaluate()

    def _get_eval_args(self, output_path, args):
        return TrainingArguments(
            output_dir=output_path,
            seed=42,
            report_to=['none'],
            per_device_eval_batch_size=args.batch_size,
            use_mps_device=True if self.device.type == "mps" else False
        )

    @staticmethod
    def action_step(ape, batch_size, gas, data_len, num_gpus) -> int:
        epoch_step_len = data_len / (batch_size * gas * num_gpus)

        action_step = math.ceil(epoch_step_len / ape)

        return action_step

    def push_to_hub(self, repo_name, private=True):
        self.logger.info("Pushing model...")
        self.model.push_to_hub(repo_name, private=private)
        self.logger.info("Pushing tokenizer...")
        self.tokenizer.push_to_hub(repo_name, private=private)

    def _init_model(self, model_type, model_name, use_auth_token):
        self.model_type = model_type
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name, use_auth_token=use_auth_token)
        self.model = self._model_class.from_pretrained(model_name, config=self.config, use_auth_token=use_auth_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
