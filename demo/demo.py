import numpy as np
from generator.multiple_sub_problems import MultiViewSubProblemsGenerator
from classify_generated import gen_folds, make_fig, test_dataset

n_views = 4
n_classes = 8
conf = np.ones((n_classes, n_views))*0.40
conf[0,3] = 0.70
conf[4,1] = 0.5
conf[5, 1] = 0.6
conf[6, 1] = 0.7
conf[7, 1] = 0.75
conf[6, 2] = 0.5
conf[7, 2] = 0.5
# conf = np.array([
#     np.array([0.40, 0.31, 0.31, 0.80]),
#     np.array([0.31, 0.31, 0.31, 0.31]),
#     np.array([0.31, 0.31, 0.31, 0.31]),
#     np.array([0.31, 0.4, 0.31, 0.31]),
#     np.array([0.31, 0.5, 0.31, 0.31]),
#     np.array([0.31, 0.6, 0.31, 0.31]),
#     np.array([0.31, 0.7, 0.41, 0.31]),
#     np.array([0.31, 0.8, 0.41, 0.31]),
# ])
n_folds = 10
n_samples = 2000
n_features = 3
class_sep = 10
class_weights = [0.125, 0.1, 0.15, 0.125, 0.01, 0.2, 0.125, 0.125,]
mutual_error = 0.1
redundancy = 0.05
complementarity = 0.5

gene = MultiViewSubProblemsGenerator(confusion_matrix=conf,
                                     n_samples=n_samples,
                                     n_views=n_views,
                                     n_classes=n_classes,
                                     class_seps=class_sep,
                                     n_features=n_features,
                                     n_informative=n_features,
                                     class_weights=class_weights,
                                     mutual_error=mutual_error,
                                     redundancy=redundancy,
                                     complementarity=complementarity)
gene.generate_multi_view_dataset()


folds = gen_folds(random_state=42, generator=gene, n_folds=n_folds)
output_confusion = test_dataset(folds, n_views, n_classes, gene)
make_fig(conf, output_confusion, n_views, n_classes, gene)