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
"""Local scalar logging for the release training loop."""
from verl.utils.logger.aggregate_logger import concat_dict_to_str


class Tracking:
    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        backends = [default_backend] if isinstance(default_backend, str) else list(default_backend)
        if backends != ["console"]:
            raise ValueError("This release writes metrics to the local console log")

    def log(self, data, step, backend=None):
        print(concat_dict_to_str(data, step=step), flush=True)
