# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Small library that points to the SUAS shapes dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset

import os
import sys
import tarfile

DATA_URL = 'azautonomous.com' # TODO 

class Shapes(Dataset):
	"""SUAS Shapes dataset"""
	def __init__(self, subset):
		super(Shapes, self).__init__('Shapes', subset)

	def num_classes(self):
		"""Returns the number of classes in the data set."""
		return 14 # 13 shapes + 1 background
		
	def meanstd(self):
		"""Returns mean and stddev of dataset as numpy array"""
		mean = [84.409, 102.408, 120.842]
		stddev = [54.420, 51.711, 58.137]
		return [mean, stddev]

	def num_examples_per_epoch(self):
		"""Returns the number of examples in the data subset."""
		if self.subset == 'train':
			return 88452
		if self.subset == 'validation':
			return 37908

	def download_data_files(self, dest_directory):
		print("Downloading data is not currently supported")
