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
"""Small library that points to the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset

import os
import sys
import tarfile
from six.moves import urllib

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

class CIFAR10(Dataset):
	"""CIFAR-10 Dataset"""
	def __init__(self, subset):
		super(CIFAR10, self).__init__('CIFAR-10', subset)

	def num_classes(self):
		"""Returns the number of classes in the data set."""
		return 10
		
	def meanstd(self):
		"""Returns mean and stddev of dataset as numpy array"""
		mean = [125.3, 123.0, 113.9] # R,G,B
		stddev = [63.0, 62.1, 66.7] # R,G,B
		return [mean, stddev]

	def num_examples_per_epoch(self):
		"""Returns the number of examples in the data subset."""
		if self.subset == 'train':
			return 50000
		if self.subset == 'validation':
			return 10000

	def download_data_files(self, dest_directory):
		"""Download and extract the tarball from Alex's website."""
		if not os.path.exists(dest_directory):
			os.makedirs(dest_directory)
		filename = DATA_URL.split('/')[-1]
		filepath = os.path.join(dest_directory, filename)
		if not os.path.exists(filepath):
			def _progress(count, block_size, total_size):
				sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
						float(count * block_size) / float(total_size) * 100.0))
				sys.stdout.flush()
			filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
			print()
			statinfo = os.stat(filepath)
			print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
		extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
		if not os.path.exists(extracted_dir_path):
			tarfile.open(filepath, 'r:gz').extractall(dest_directory)
