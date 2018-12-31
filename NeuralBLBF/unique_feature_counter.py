"""
This script calculates number of the unique categories
"""

from collections import Counter
from tqdm import tqdm

prop_file = open("propfile.txt", "w")

DATA_TRAIN = '../data/vw_compressed_train'
DATA_TEST = '../data/vw_compressed_test'
DATA_VALIDATE = '../data/vw_compressed_validate'

DATA = [DATA_TRAIN, DATA_TEST, DATA_VALIDATE]

CNT = Counter()
max_product_features = 0
min_product_features = 100000
count = 0

for sub_data in DATA:
	sample_counter = 0
	with open(sub_data) as f:
		for line in tqdm(f):
			count += 1
			line = line.strip()
			if "shared" in line:
				sample_counter += 1
				count = 0
				line = ' '.join(line.split("|")[1].split()[2:])
			elif '|' in line:
				if count == 1:
					[propensity, line] = line.split("|")
					[_, click, propensity] = propensity.split(":")
					prop_file.write(propensity + " " + str(round(float(click))) + "\n")
				else:
					line = line.split("|")[-1]
				num = len(line.split())
				if num > max_product_features:
					max_product_features = num
				elif num < min_product_features:
					min_product_features = num
			for feature in line.split():
				if ":" in feature:
					feature = feature.split(":")[0]
				CNT[feature] += 1
	print("\n" + "length features counter: " + str(len(CNT.keys())))
	print("sample counter: " + str(sample_counter) + "\n")
	prop_file.write(":" + "\n")

prop_file.close()
print("min features: " + str(min_product_features))
print("max featrues: " + str(max_product_features))
feature_file = open("featurefile.txt", "w")
for k, v in CNT.most_common():
	feature_file.write("{} {}\n".format(k, v))
feature_file.close()

