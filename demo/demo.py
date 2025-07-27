import numpy as np
from multiview_generator.gaussian_classes import MultiViewGaussianSubProblemsGenerator
from classify_generated import gen_folds, make_fig, test_dataset

n_views = 4
n_classes = 3
gene = MultiViewGaussianSubProblemsGenerator(config_file="config_demo.yml")
conf = np.ones((n_classes, n_views))*0.4
gene.generate_multi_view_dataset()
gene.to_hdf5_mc()
print(gene.gen_report())

folds = gen_folds(random_state=42, generator=gene)
output_confusion = test_dataset(folds, n_views, n_classes, gene)
make_fig(conf, output_confusion, n_views, n_classes, gene)