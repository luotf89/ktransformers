from mmlu_test import DataEvaluator
import argparse
import random
import time
import json
import requests
import pandas as pd
from datasets import load_dataset
import os
data_evaluator = DataEvaluator()
data_evaluator.load_data("/home/ltf/mmlu/")
random.seed(42)
random.shuffle(data_evaluator.data)
for i in range(min(1000, len(data_evaluator.data))):
    # Randomly select a data item from data for each request
    data_item = data_evaluator.data[i]
    question = data_evaluator.get_prompt(data_item)
    print (f"id:{i},chat:{question}")
