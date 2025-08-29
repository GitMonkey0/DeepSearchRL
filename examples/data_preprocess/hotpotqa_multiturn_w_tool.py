# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the FlashRAG QA dataset to parquet format (train & test)
"""

import argparse
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs


def make_prefix(dp, template_type):
    question = dp['question']

    if template_type == 'base':
        prefix = (
            f"Answer the question step by step. "
            f"You may call retrieve_documents whenever you need more information. "
            f"When you are ready, put the final answer between <answer> and </answer>, "
            f"e.g. <answer>Beijing</answer>. "
            f"Question: {question}\n"
        )
    else:
        raise NotImplementedError
    return prefix


def build_split(data_source: str, split: str, template_type: str):
    """Load one split and map it to the common format."""
    ds = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)[split]

    if split == 'train':
        ds = ds.select(range(min(51200, len(ds))))

    def process_fn(example, idx):
        example['question'] = example['question'].strip()
        prefix = make_prefix(example, template_type=template_type)
        data = {
            "data_source": f"searchR1_{data_source}",
            "prompt": [{"role": "user", "content": prefix}],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": example['golden_answers']}
            },
            "extra_info": {
                'split': split,
                'index': idx,
                "tools_kwargs": {
                    "retrieve_documents": {
                        "create_kwargs": {"": ""},
                    }
                },
            },
        }
        return data

    return ds.map(process_fn, with_indices=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/hotpotqa")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--template_type", type=str, default="base")
    parser.add_argument("--data_sources", default="hotpotqa")

    args = parser.parse_args()
    data_sources = args.data_sources.split(',')

    train_splits, test_splits = [], []
    for src in data_sources:
        train_splits.append(build_split(src, 'train', args.template_type))

        for cand in ['test', 'dev', 'train']:
            if cand in datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', src):
                test_splits.append(build_split(src, cand, args.template_type))
                break

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    datasets.concatenate_datasets(train_splits).to_parquet(
        os.path.join(local_dir, "train.parquet")
    )
    datasets.concatenate_datasets(test_splits).to_parquet(
        os.path.join(local_dir, "test.parquet")
    )

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
