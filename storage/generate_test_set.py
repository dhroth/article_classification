from collections import Counter
import random
from storage import article_store

article_ids = list(article_store.get_all_article_ids())
random.shuffle(article_ids)

#number of test items for each class
num_desired_test_items = 1000

num_test_items = Counter()
test_items = []
for article_id in article_ids:
    try:
        version = article_store.get_version(article_id)
        if version == article_store.PUBLISHER_ID:
            if num_test_items["publisher"] < num_desired_test_items:
                test_items.append(article_id)
                num_test_items["publisher"] += 1
        elif version == article_store.MANUSCRIPT_ID:
            if num_test_items["manuscript"] < num_desired_test_items:
                test_items.append(article_id)
                num_test_items["manuscript"] += 1
    except RuntimeError as e:
        pass

with open("storage/test_set.csv", "w") as test_set_file:
    for article_id in test_items:
        test_set_file.write(article_id + "\n")
