"""
This script calculates number of the unique caterogies
"""

from collections import Counter
from tqdm import tqdm

DATA_TRAIN = 'data/vw_compressed_train'
DATA_TEST = 'data/vw_compressed_test'
DATA_VALIDATE = 'data/vw_compressed_validate'

DATA = [DATA_TRAIN, DATA_TEST, DATA_VALIDATE]

CNT = Counter()
for sub_data in DATA:
	with open(sub_data) as f:
		count = 0
		for line in tqdm(f):
			count += 1
			line = line.strip()
			if "shared" in line:
				line = ' '.join(line.split("|")[1].split()[2:])
				count = 0
			else:
				line = line.split("|")[1]
			for feature in line.split():
				if ":" in feature:
					feature = feature.split(":")[0]
				CNT[feature] += 1
	print(len(CNT.keys()))
print(len(CNT.keys()))
